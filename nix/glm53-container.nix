{ config, lib, pkgs, ... }:

# NixOS module: vLLM GLM-5.3-Flash-nvfp4, pipeline-parallel PP=5 across all
# five CMP 170HX GPUs. Port 8000, model name "glm53flash".
#
# Import in configuration.nix:
#   imports = [ ... ./vllm-170hx-glm5.3-flash/nix/glm53-container.nix ... ];
#
# Expects:
#   - this repo checked out at /opt/vllm-170hx-glm5.3-flash (or adjust the
#     override paths below)
#   - the model at /models/GLM-5.3-Flash-nvfp4
#     (hf download mbehr90/GLM-5.3-Flash-nvfp4 --local-dir /models/GLM-5.3-Flash-nvfp4)
#   - podman + NVIDIA CDI device wiring (nvidia.com/gpu=all)
#
# NOTE: this unit owns GPUs 0-4. Disable conflicting GPU services
# (e.g. dsv4-serve, vision stacks) before enabling it.
{
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
        "docker.io/vllm/vllm-openai:glm53-flash"
        "/model"
        "--served-model-name glm53flash"
        "--port 8000"
        "--pipeline-parallel-size 5"
        "--moe-backend humming"
        "--safetensors-load-strategy eager"
        # 1M = max_position_embeddings of the model (true maximum). Dense
        # attention: prefill of huge prompts is slow (chunked at 8192 tok/step)
        # and decode slows near the ceiling.
        "--max-model-len 1048576"
        # Split the model's always-on thinking (template auto-opens <think>)
        # into reasoning_content. glm47 parser starts in REASONING state and
        # switches on </think> — matches this template exactly.
        "--reasoning-parser glm47"
        "--gpu-memory-utilization 0.92"
        "--max-num-seqs 64"
        # Agent harnesses send tool_choice=auto — rejected without these.
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
