# vLLM GLM-5.3-Flash (nvfp4) on 5× CMP 170HX — pipeline-parallel PP=5

Run [mbehr90/GLM-5.3-Flash-nvfp4](https://huggingface.co/mbehr90/GLM-5.3-Flash-nvfp4)
(191 GiB compressed-tensors NVFP4A16, 321B-param MoE) with vLLM
**pipeline parallelism across five GPUs** on CMP 170HX mining cards
(GA100 / sm_80 / 64 GiB after unlock).

The model's stock serving image (`vllm/vllm-openai:glm53-flash`) assumes
Hopper-class GPUs and no pipeline parallelism. This repo carries the seven
override files that make it run on Ampere with PP=5.

**Measured on the reference machine (5× unlocked 170HX): ~43 tok/s
single-stream decode, 32k context, ~0.5 s first token, ~20 min model load.**

## Hardware requirements

- 5× CMP 170HX unlocked to 64 GiB — [cmpunlocker](https://github.com/amoghmunikote/cmpunlocker)
  with the patched `nvidia-open` 610.43.03 driver (see that repo for NixOS
  wiring: IOMMU passthrough, Secure Boot off, Gen2 retrain hammer).
  Any 5× 64 GiB Ampere GPUs (A100-class, sm_80) should work equally.
- ~42 GiB weights per GPU + ~10 GiB KV cache per GPU at `--gpu-memory-utilization 0.90`.
- PCIe topology does not matter much (PP, not TP — no all-reduce per layer),
  but expect the weight load to be I/O bound (~20 min, serialized).

## Download the model

```bash
hf download mbehr90/GLM-5.3-Flash-nvfp4 --local-dir /models/GLM-5.3-Flash-nvfp4
# 191 GiB, 48 safetensors shards
```

## Quickstart

### podman / docker (bind-mount launcher — the 1:1 validated setup)

```bash
MODEL_DIR=/models/GLM-5.3-Flash-nvfp4 ./run.sh
```

### docker compose (builds the derived image with patches baked in)

```bash
docker compose up -d --build
```

Adjust the model path in `docker-compose.yml`. The derived image bakes the
six vLLM code overrides; the `config.json` overlay always stays a mount on
top of the model directory.

### NixOS

```nix
# configuration.nix
imports = [ /path/to/this/repo/nix/glm53-container.nix ];
```

Expects this repo at `/opt/vllm-170hx-glm5.3-flash` and the model at
`/models/GLM-5.3-Flash-nvfp4` (adjust paths in the module otherwise).
This unit owns **all five GPUs** — disable conflicting GPU services first.

### Use it

```bash
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm53flash",
  "max_tokens": 128,
  "messages": [{"role": "user", "content": "hello"}]
}'
```

## The seven overrides

| File | Mounts over | Why |
|---|---|---|
| `overrides/model_patched.py` | `vllm/models/glm5next/nvidia/model.py` | **PP unlock**: `make_empty_intermediate_tensors` with the hc-expanded `[tokens, 4, hidden]` residual (stock image has none — PP gated off); **mHC rank-boundary fix** (stock `hc_expand` only ran at `layer_idx 0`, ranks > 0 must reuse the incoming 3-D residual); dense-mode indexer loader guards; flock-serialized per-rank weight loading |
| `overrides/overlay/config.json` | `/model/config.json` | `index_topk: null` — disables the DSA sparse indexer → **dense MLA via TRITON_MLA** (every sparse-MLA backend in the image is Hopper+). Dense = full attention, the mathematically complete model |
| `overrides/overlay/triton_mla_prefill_sm80.py` | new file in `.../mla/prefill/` | Custom **Triton ragged MLA prefill backend** for the model's NoPE dims (qk_nope=256, rope=0, v=256): varlen, separate q/k ragged offsets for context chunks, softmax-LSE out (`[heads, total_q]`), 128×64 tiles to fit GA100's 164 KiB smem at head-dim 256 |
| `overrides/selector_patched.py` | `.../mla/prefill/selector.py` | Registers the backend above as the `CUSTOM` MLA prefill backend and prefers it below sm90 (stock only offers FA there, which rejects these dims) |
| `overrides/weight_utils_patched.py` | `.../model_loader/weight_utils.py` | safetensors iterator with a numpy-framework fallback for `get_tensor` (BTRFS mmap reads in workers occasionally return corrupt storages) |
| `overrides/gpu_worker_patched.py` | `vllm/v1/worker/gpu_worker.py` | `set_device` retry + `VLLM_WORKER_INIT_STAGGER_S`: concurrent 5-worker CUDA context creation sporadically fails on this platform |
| `overrides/marlin_f4_patched.py` | `.../quantization/utils/marlin_utils_fp4.py` | Marlin nvfp4 prep with the scale pipeline on CPU + flock + entry sync. **Inert while `--moe-backend humming`**; kept for when the marlin path is fixed (see `lab/NOTES.md`) |

## Serving flags that matter

- `--moe-backend humming` — Triton fp4 experts. **Do not** switch to `marlin`
  on a 5-rank setup before reading `lab/NOTES.md` (async illegal accesses
  during weight prep; single-process marlin prep is clean).
- `--safetensors-load-strategy eager` — avoids mmap reads (BTRFS corruption);
  costs ~5 s/shard.
- `--pipeline-parallel-size 5` — 45 decoder layers → 9 per rank.
- `--max-model-len 32768`, `--max-num-seqs 64`, `--gpu-memory-utilization 0.90`.
- `--enable-auto-tool-choice --tool-call-parser glm47` — required by agent
  harnesses sending `tool_choice: "auto"`; `glm47` is the GLM-family parser
  in this image (registered as `glm45`/`glm47`). If tool calls come out
  malformed, compare the model's actual markup against the parser.
- No MTP/speculative decoding: the model's nextn draft layer (layer 45) is
  skipped at load (`speculative_config=None`).

## Operational notes & known issues

- **~20 min model load**, serialized per rank (PCIe-stability measure), then
  ~1 min CUDA-graph capture. Readiness line: `Application startup complete.`
- **A GPU crash can wedge the driver**: nvidia-smi keeps looking healthy, but
  the next start fails `cudaSetDevice` (busy/unavailable or OOM) on the wedged
  GPU. Only a host reboot recovers — a restart-looping service is the tell.
- The model emits reasoning-style text into `content`; wire a reasoning parser
  if you want it split out.
- On the reference NixOS box this stack displaces the DeepSeek-V4 and
  vision-serving services (they own overlapping GPUs).

## Lab

`lab/` contains the single-process probes used to exonerate the marlin path
during the crash investigation, plus `lab/NOTES.md` — the full root-cause log
and the offline pre-repack upgrade path for Marlin-grade throughput.

## License

MIT. The model is MIT-licensed (see its card); vLLM is Apache-2.0.
