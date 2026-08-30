#!/usr/bin/env bash
# GLM-5.3-Flash-nvfp4 on 5x CMP 170HX via vowstar/vllm-sm80 (branch glm53-sm80).
#
# Alternative stack to run.sh (stock image + overrides): a source fork with
# native sm_80 support — sparse MLA (DSA indexer) via Triton, fp8 latent KV,
# marlin W4A16 MoE, MTP x3 spec decode, prefix caching. Validated upstream at
# 1M context on identical hardware (5x CMP 170HX, PP5).
#
# Image build (once, ~1 h on 64 cores):
#   cd /root/vllm-sm80 && podman build \
#     --build-arg torch_cuda_arch_list="8.0" --build-arg max_jobs=40 \
#     -t vllm-sm80:latest -f docker/Dockerfile .
#
# Usage: MODEL_DIR=/path/to/model ./run-sm80.sh   (SPEC_MTP=1 to enable MTP x3; needs a
#   checkpoint with a quantized draft layer — mbehr90's has a bf16 draft and
#   crashes the marlin backend)
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models/GLM-5.3-Flash-nvfp4}"
CONTAINER_NAME="${CONTAINER_NAME:-glm53-sm80}"

if command -v podman >/dev/null 2>&1; then
  ENGINE=podman
else
  ENGINE=docker
fi

"$ENGINE" rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

# shellcheck disable=SC2086
exec "$ENGINE" run -d \
  --name "$CONTAINER_NAME" \
  --device nvidia.com/gpu=all \
  --network=host \
  --shm-size=16g \
  --cap-add IPC_OWNER \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=7200 \
  -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600 \
  -e NCCL_P2P_DISABLE=1 \
  -e TORCH_SHOW_CPP_STACKTRACES=1 \
  -e VLLM_PP_LAYER_PARTITION=11,9,9,9,7 \
  -e VLLM_PREFIX_CACHE_RETENTION_INTERVAL="${VLLM_PREFIX_CACHE_RETENTION_INTERVAL:-139264}" \
  -e VLLM_MARLIN_REPACK_HOLDOFF=1 \
  -v "${MODEL_DIR}:/model:ro" \
  vllm-sm80:latest \
  /model \
    --served-model-name glm53flash \
    --port 8000 \
    --pipeline-parallel-size 5 \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --max-model-len "${MAX_MODEL_LEN:-1048576}" \
    --max-num-batched-tokens 8192 \
    --trust-remote-code \
    --gpu-memory-utilization 0.89 \
    --max-num-seqs 32 \
    --kv-cache-memory 13421772800 \
    --reasoning-parser glm47 \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --moe-backend marlin \
    ${SPEC_MTP:+--speculative-config '{"method":"mtp","num_speculative_tokens":3}'}

echo "container: ${CONTAINER_NAME} — follow: ${ENGINE} logs -f ${CONTAINER_NAME}"
