# Derived image: stock vllm/vllm-openai:glm53-flash with the six vLLM code
# overrides baked in. The config.json overlay cannot be baked (the model
# directory is a runtime mount) — mount it alongside the model:
#   -v ./overrides/overlay/config.json:/model/config.json:ro
FROM vllm/vllm-openai:glm53-flash

COPY overrides/model_patched.py /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py
COPY overrides/selector_patched.py /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/prefill/selector.py
COPY overrides/overlay/triton_mla_prefill_sm80.py /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/prefill/triton_mla_prefill_sm80.py
COPY overrides/weight_utils_patched.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py
COPY overrides/gpu_worker_patched.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py
COPY overrides/marlin_f4_patched.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py
