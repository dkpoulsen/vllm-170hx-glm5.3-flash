{ config, lib, pkgs, ... }:

{
  # ---------------------------------------------------------------------------
  # vLLM model server — GLM-5.3-Flash-nvfp4 via vowstar/vllm-sm80
  # (branch glm53-sm80, built locally as localhost/vllm-sm80:latest).
  # Serving on ALL 5 CMP 170HX GPUs, PP=5, port 8000, model "glm53flash".
  #
  # This is the PREFERRED stack (2026-08-30): 1,048,576-token context with
  # zero Xids (wall 2 does not exist here), ~95 s model load, 7.49M-token
  # KV pool, ~44 tok/s decode. Replaces the stock-image stack in
  # glm53-container.nix (kept as opt-in fallback, 49152 cap).
  #
  # One-time image build (~2 h on 64 cores, see repo run-sm80.sh header):
  #   cd /tmp/vllm-sm80 && podman build \
  #     --build-arg torch_cuda_arch_list="8.0" --build-arg max_jobs=40 \
  #     -t vllm-sm80:latest -f docker/Dockerfile .
  #
  # Serving shape (completely different from the stock-image stack):
  # sparse MLA (DSA indexer, index_topk=2048 from the stock model config)
  # via the fork's Triton sm_80 kernel, fp8 e4m3 latent KV (software
  # dequant in-kernel), marlin W4A16 MoE, block 256, layer partition
  # 11,9,9,9,7 (last rank carries lm_head + MTP draft).
  #
  # Gotchas baked into the flags below:
  #   - image entrypoint already execs `vllm serve` -> CMD is `/model ...`
  #   - VLLM_PREFIX_CACHE_RETENTION_INTERVAL must be a multiple of OUR
  #     scheduler_block_size 8704 (their README's 143360 assumes 8960)
  #     -> 139264 = 16 x 8704
  #   - NO MTP spec decode: the mbehr90 checkpoint stores the layer-45
  #     draft experts UNQUANTIZED -> UnquantizedFusedMoEMethod rejects
  #     moe_backend=marlin at draft init on PP4. (Their ~100 tok/s decode
  #     needs MTP; ours ~44 tok/s.)
  #   - VLLM_MARLIN_REPACK_HOLDOFF is flagged unknown by this build but is
  #     the upstream recommendation against a load-time MMU fault; inert
  #     here, kept for safety.
  #
  # GPU ownership: needs GPUs 0-4; conflicts with glm53-serve (starting
  # either stops the other). dsv4-serve / qwen-ninfer / fusion-router
  # stay opt-in (their wantedBy is commented out).
  #
  # Same wedge warning as glm53-serve: a GPU crash can wedge the RM driver
  # (nvidia-smi still looks healthy) — only a host reboot recovers.
  # ---------------------------------------------------------------------------

  systemd.services.glm53-sm80-serve = {
    description = "vLLM GLM-5.3-Flash-nvfp4 PP5 container — vowstar vllm-sm80 (1M ctx)";
    wants = [ "network-online.target" ];
    after = [ "network-online.target" ];
    wantedBy = [ "multi-user.target" ];
    conflicts = [ "glm53-serve.service" ];

    serviceConfig = {
      Type = "notify";
      NotifyAccess = "all";
      Restart = "on-failure";
      RestartSec = 30;
      TimeoutStartSec = 3600;
      Environment = "PODMAN_SYSTEMD_UNIT=%n";
      ExecStart = lib.concatStringsSep " \\\n  " [
        "${pkgs.podman}/bin/podman run"
        "--cidfile=/run/glm53-sm80-serve.ctr-id"
        "--cgroups=no-conmon"
        "--rm"
        "--sdnotify=conmon"
        "--replace"
        "-d"
        "--name glm53-sm80"
        "--device nvidia.com/gpu=all"
        "--network=host"
        "--shm-size=16g"
        "--cap-add IPC_OWNER"
        "-e HF_HUB_OFFLINE=1"
        "-e VLLM_ENGINE_READY_TIMEOUT_S=7200"
        "-e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600"
        "-e NCCL_P2P_DISABLE=1"
        "-e TORCH_SHOW_CPP_STACKTRACES=1"
        # Even layer split by checkpoint BYTES (11,9,9,9,7): the last rank
        # also carries lm_head and the MTP draft layer.
        "-e VLLM_PP_LAYER_PARTITION=11,9,9,9,7"
        # 16 mamba-aligned pages of OUR scheduler_block_size 8704 (fp8 KV).
        "-e VLLM_PREFIX_CACHE_RETENTION_INTERVAL=139264"
        "-e VLLM_MARLIN_REPACK_HOLDOFF=1"
        "-v /models/GLM-5.3-Flash-nvfp4:/model:ro"
        "localhost/vllm-sm80:latest"
        # Entrypoint already execs `vllm serve` — do not repeat it.
        "/model"
        "--served-model-name glm53flash"
        "--port 8000"
        "--pipeline-parallel-size 5"
        "--kv-cache-dtype fp8"
        # Indexer cache needs block_size % 128 == 0 and the Triton sparse
        # MLA kernel declares MultipleOf(64); 256 satisfies both.
        "--block-size 256"
        # Model max. Validated 2026-08-30: 1,039k-token prompts clean,
        # 205 s cold prefill, 7 s cached appends, zero Xids.
        "--max-model-len 1048576"
        "--max-num-batched-tokens 8192"
        "--trust-remote-code"
        "--gpu-memory-utilization 0.89"
        "--max-num-seqs 32"
        # The 0.89-derived pool is too small: at saturation the mamba state
        # copies evict all hashed checkpoints (upstream-validated value).
        "--kv-cache-memory 13421772800"
        # glm47 parser works in this fork for TOOL calls; reasoning_content
        # still comes back empty (same as the stock stack's glm47 adapter).
        "--reasoning-parser glm47"
        "--enable-auto-tool-choice"
        "--tool-call-parser glm47"
        "--moe-backend marlin"
      ];
      ExecStop = "${pkgs.podman}/bin/podman stop --ignore -t 10 --cidfile=/run/glm53-sm80-serve.ctr-id";
      ExecStopPost = "${pkgs.podman}/bin/podman rm -f --ignore -t 10 --cidfile=/run/glm53-sm80-serve.ctr-id";
    };
  };

  networking.firewall.allowedTCPPorts = lib.mkBefore [ 8000 ];
}
