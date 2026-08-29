{ config, lib, pkgs, ... }:

{
  # ---------------------------------------------------------------------------
  # vLLM model server — GLM-5.3-Flash-nvfp4 (mbehr90, compressed-tensors
  # NVFP4A16) serving on ALL 5 CMP 170HX GPUs with pipeline parallelism PP=5.
  # Port 8000, model name "glm53flash", ~43 tok/s single-stream @ 32k ctx.
  #
  # Image: vllm/vllm-openai:glm53-flash (arch Glm5NextForConditionalGeneration,
  # v0.1.dev20051+g487ecf187 — not in any released vLLM).
  #
  # The fourteen bind-mounted override files under /opt/vllm-170hx-glm5.3-flash/overrides/
  # REQUIRED — the stock image cannot run this model on sm_80 / with PP:
  #   model_patched.py            -> glm5next model.py: PP unlock
  #                                  (make_empty_intermediate_tensors with the
  #                                  hc-expanded [tokens, 4, hidden] residual),
  #                                  mHC rank-boundary fix, dense-mode indexer
  #                                  loader guards, flock-serialized loading.
  #   overlay/config.json         -> /model/config.json overlay: index_topk=null
  #                                  (kills the DSA sparse indexer -> dense MLA
  #                                  via TRITON_MLA; sparse MLA is Hopper+).
  #   overlay/triton_mla_prefill_sm80.py
  #                                -> custom Triton ragged MLA prefill backend
  #                                  (NoPE 256/0/256 dims; FA path is dims-
  #                                  gated, FlashInfer/TRT-LLM are sm90/100).
  #   selector_patched.py         -> MLA prefill selector: registers the CUSTOM
  #                                  backend above, prefers it below sm90.
  #   weight_utils_patched.py     -> safetensors iterator with numpy-framework
  #                                  fallback (BTRFS mmap corruption guard).
  #   gpu_worker_patched.py       -> set_device retry + worker init stagger
  #                                  (concurrent 5-worker context creation is
  #                                  flaky on this box).
  #   marlin_f4_patched.py        -> marlin nvfp4 prep with CPU scale pipeline
  #                                  + flock (INERT while --moe-backend
  #                                  humming; needed if backend flips back).
  #
  # --moe-backend humming, not marlin: the nvfp4 marlin prep crashes with async
  # illegal accesses inside the 5-worker engine (exonerated single-process —
  # see the 170HX skill llm-stack.md). Suspected PCIe x4 link instability
  # under 5-way load DMA (chronic AER Data-Link-Layer replay timeouts).
  # Marlin-grade upgrade path: offline single-proc pre-repack + loader patch.
  #
  # GPU ownership: this unit needs GPUs 0-4. dsv4-serve (vllm-container.nix),
  # qwen-ninfer + fusion-router (fusion.nix) have their wantedBy commented out
  # while this stack owns the box. Restore those to swap back.
  #
  # WARNING: any GPU crash can wedge the RM driver (stuck memory scrubber,
  # nvidia-smi still looks healthy) — only a host reboot recovers; the service
  # will restart-loop uselessly until then.
  #
  # 2026-08-28: first boot of this module, ported from the validated ad-hoc
  # launcher /root/glm53-nvfp4-work/launch-pp5.sh (kept as the lab copy).
  # ---------------------------------------------------------------------------

  systemd.services.glm53-serve = {
    description = "vLLM GLM-5.3-Flash-nvfp4 PP5 container (all 5 CMP 170HX)";
    wants = [ "network-online.target" ];
    after = [ "network-online.target" ];
    wantedBy = [ "multi-user.target" ];

    serviceConfig = {
      Type = "notify";
      NotifyAccess = "all";
      Restart = "on-failure";
      RestartSec = 30;
      TimeoutStartSec = 3600;
      Environment = "PODMAN_SYSTEMD_UNIT=%n";
      ExecStart = lib.concatStringsSep " \\\n  " [
        "${pkgs.podman}/bin/podman run"
        "--cidfile=/run/glm53-serve.ctr-id"
        "--cgroups=no-conmon"
        "--rm"
        "--sdnotify=conmon"
        "--replace"
        "-d"
        "--name glm53-pp5"
        "--device nvidia.com/gpu=all"
        "--network=host"
        "--shm-size=16g"
        "--cap-add IPC_OWNER"
        "-e HF_HUB_OFFLINE=1"
        "-e VLLM_WORKER_MULTIPROC_METHOD=spawn"
        # Serialized load (flock) + eager reads: full model load takes ~20 min.
        "-e VLLM_ENGINE_READY_TIMEOUT_S=7200"
        "-e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600"
        "-e NCCL_P2P_DISABLE=1"
        "-e VLLM_MARLIN_LOCK=/dev/shm/marlin_prep.lock"
        "-e VLLM_WORKER_INIT_STAGGER_S=15"
        "-e VLLM_MARLIN_SCALES_ON_CPU=1"
        "-e TORCH_SHOW_CPP_STACKTRACES=1"
        "-v /opt/vllm-170hx-glm5.3-flash/overrides/gpu_worker_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py:ro"
        "-v /opt/vllm-170hx-glm5.3-flash/overrides/marlin_f4_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py:ro"
        "-v /models/GLM-5.3-Flash-nvfp4:/model:ro"
        "-v /opt/vllm-170hx-glm5.3-flash/overrides/overlay/config.json:/model/config.json:ro"
        "-v /opt/vllm-170hx-glm5.3-flash/overrides/model_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py:ro"
        "-v /opt/vllm-170hx-glm5.3-flash/overrides/overlay/triton_mla_prefill_sm80.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/prefill/triton_mla_prefill_sm80.py:ro"
        "-v /opt/vllm-170hx-glm5.3-flash/overrides/selector_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/prefill/selector.py:ro"
        "-v /opt/vllm-170hx-glm5.3-flash/overrides/weight_utils_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py:ro"
        # KDA/FLA prefill kernels with int64 index math — stock kernels
        # overflow int32 above 262144 tokens (token*8192 element offsets),
        # crashing with Xid 31 + driver wedge. Patched sites: cu_seqlens
        # loads (bos/eos), T length cast, absolute gate row, per-chunk h-state
        # indexing. Validated: short-run bit-identical to stock, 270k-token
        # single-sequence run clean.
        "-v /opt/vllm-170hx-glm5.3-flash/overrides/fla/kda.py:/usr/local/lib/python3.12/dist-packages/vllm/third_party/flash_linear_attention/ops/kda.py:ro"
        "-v /opt/vllm-170hx-glm5.3-flash/overrides/fla/chunk_o.py:/usr/local/lib/python3.12/dist-packages/vllm/third_party/flash_linear_attention/ops/chunk_o.py:ro"
        "-v /opt/vllm-170hx-glm5.3-flash/overrides/fla/chunk_delta_h.py:/usr/local/lib/python3.12/dist-packages/vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py:ro"
        "-v /opt/vllm-170hx-glm5.3-flash/overrides/fla/chunk_scaled_dot_kkt.py:/usr/local/lib/python3.12/dist-packages/vllm/third_party/flash_linear_attention/ops/chunk_scaled_dot_kkt.py:ro"
        "-v /opt/vllm-170hx-glm5.3-flash/overrides/fla/cumsum.py:/usr/local/lib/python3.12/dist-packages/vllm/third_party/flash_linear_attention/ops/cumsum.py:ro"
        "-v /opt/vllm-170hx-glm5.3-flash/overrides/fla/solve_tril.py:/usr/local/lib/python3.12/dist-packages/vllm/third_party/flash_linear_attention/ops/solve_tril.py:ro"
        "-v /opt/vllm-170hx-glm5.3-flash/overrides/fla/wy_fast.py:/usr/local/lib/python3.12/dist-packages/vllm/third_party/flash_linear_attention/ops/wy_fast.py:ro"
        "docker.io/vllm/vllm-openai:glm53-flash"
        "/model"
        "--served-model-name glm53flash"
        "--port 8000"
        "--pipeline-parallel-size 5"
        "--moe-backend humming"
        "--safetensors-load-strategy eager"
        # 1M = model max. Safe again: the KDA/FLA prefill kernels were patched
        # to int64 index math (see the fla/ overrides) — stock kernels
        # overflowed int32 above 262144 tokens (Xid 31 MMU fault + wedge).
        "--max-model-len 1048576"
        # Split the model's always-on thinking (template auto-opens <think>)
        # into reasoning_content. NOTE: deepseek_r1, NOT glm47 — the fork's
        # glm47 reasoning adapter silently swallows thinking (never emits
        # reasoning_content, drops unterminated reasoning on truncation);
        # deepseek_r1 explicitly handles output that starts already inside
        # <think> (opening tag in the prompt), streaming reasoning deltas
        # until </think>. Staged 2026-08-28; applies at next rebuild/restart.
        "--reasoning-parser deepseek_r1"
        "--gpu-memory-utilization 0.92"
        "--max-num-seqs 64"
        # Agent harnesses (dsh) send tool_choice=auto — rejected without these.
        # glm47 = Glm47MoeModelToolParser, the GLM-family tool format in this
        # fork (registered as glm45/glm47).
        "--enable-auto-tool-choice"
        "--tool-call-parser glm47"
      ];
      ExecStop = "${pkgs.podman}/bin/podman stop --ignore -t 10 --cidfile=/run/glm53-serve.ctr-id";
      ExecStopPost = "${pkgs.podman}/bin/podman rm -f --ignore -t 10 --cidfile=/run/glm53-serve.ctr-id";
    };
  };

  networking.firewall.allowedTCPPorts = lib.mkBefore [ 8000 ];
}
