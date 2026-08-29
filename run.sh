#!/usr/bin/env bash
# GLM-5.3-Flash-nvfp4 on 5x CMP 170HX (sm_80 / GA100) — vLLM pipeline-parallel PP=5.
#
# 1:1 reproduction of the validated setup: stock vllm/vllm-openai:glm53-flash
# image with the seven override files bind-mounted over it, no image build.
#
# Requirements:
#   - 5x CMP 170HX unlocked to 64 GiB (cmpunlocker + patched nvidia-open),
#     or 5x any 64 GiB Ampere GPU (sm_80). ~42 GiB weights per GPU.
#   - docker or podman with the NVIDIA container toolkit (CDI).
#   - The model downloaded (191 GiB):
#       hf download mbehr90/GLM-5.3-Flash-nvfp4 --local-dir /models/GLM-5.3-Flash-nvfp4
#
# Usage: MODEL_DIR=/path/to/model ./run.sh
set -euo pipefail

OVERRIDES="$(cd "$(dirname "$0")" && pwd)/overrides"
MODEL_DIR="${MODEL_DIR:-/models/GLM-5.3-Flash-nvfp4}"
CONTAINER_NAME="${CONTAINER_NAME:-glm53-pp5}"

if command -v podman >/dev/null 2>&1; then
  ENGINE=podman
else
  ENGINE=docker
fi

VLLM_PY=/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py
PREFILL_DIR=/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/prefill
FLA_DIR=/usr/local/lib/python3.12/dist-packages/vllm/third_party/flash_linear_attention/ops

"$ENGINE" rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

# shellcheck disable=SC2086
exec "$ENGINE" run -d \
  --name "$CONTAINER_NAME" \
  --device nvidia.com/gpu=all \
  --network=host \
  --shm-size=16g \
  --cap-add IPC_OWNER \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_ENGINE_READY_TIMEOUT_S=7200 \
  -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600 \
  -e NCCL_P2P_DISABLE=1 \
  -e VLLM_MARLIN_LOCK=/dev/shm/marlin_prep.lock \
  -e VLLM_WORKER_INIT_STAGGER_S=15 \
  -e VLLM_MARLIN_SCALES_ON_CPU=1 \
  -e TORCH_SHOW_CPP_STACKTRACES=1 \
  -v "${OVERRIDES}/gpu_worker_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py:ro" \
  -v "${OVERRIDES}/marlin_f4_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py:ro" \
  -v "${MODEL_DIR}:/model:ro" \
  -v "${OVERRIDES}/overlay/config.json:/model/config.json:ro" \
  -v "${OVERRIDES}/model_patched.py:${VLLM_PY}:ro" \
  -v "${OVERRIDES}/overlay/triton_mla_prefill_sm80.py:${PREFILL_DIR}/triton_mla_prefill_sm80.py:ro" \
  -v "${OVERRIDES}/selector_patched.py:${PREFILL_DIR}/selector.py:ro" \
  -v "${OVERRIDES}/weight_utils_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py:ro" \
  -v "${OVERRIDES}/fla/kda.py:${FLA_DIR}/kda.py:ro" \
  -v "${OVERRIDES}/fla/chunk_o.py:${FLA_DIR}/chunk_o.py:ro" \
  -v "${OVERRIDES}/fla/chunk_delta_h.py:${FLA_DIR}/chunk_delta_h.py:ro" \
  -v "${OVERRIDES}/fla/chunk_scaled_dot_kkt.py:${FLA_DIR}/chunk_scaled_dot_kkt.py:ro" \
  -v "${OVERRIDES}/fla/cumsum.py:${FLA_DIR}/cumsum.py:ro" \
  -v "${OVERRIDES}/fla/solve_tril.py:${FLA_DIR}/solve_tril.py:ro" \
  -v "${OVERRIDES}/fla/wy_fast.py:${FLA_DIR}/wy_fast.py:ro" \
  vllm/vllm-openai:glm53-flash \
  /model \
    --served-model-name glm53flash \
    --port 8000 \
    --pipeline-parallel-size 5 \
    --moe-backend humming \
    --max-model-len 1048576 \
    --reasoning-parser deepseek_r1 \
    --safetensors-load-strategy eager \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 64 \
    --enable-auto-tool-choice \
    --tool-call-parser glm47

echo "container: ${CONTAINER_NAME} — follow: ${ENGINE} logs -f ${CONTAINER_NAME}"
echo "first load takes ~20 min (serialized per-rank loading); ready line:"
echo "  \"Application startup complete.\""
