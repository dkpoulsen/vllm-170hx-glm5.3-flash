# Lab notes — the marlin crash campaign (2026-08-28)

Root-cause log for why this stack runs `--moe-backend humming` instead of the
(faster) Marlin nvfp4 path. All probes in this directory reproduce pieces of
the investigation against the stock `vllm/vllm-openai:glm53-flash` image.

## Symptom

With `--moe-backend marlin`, the 5-worker PP=5 engine crashes during
`process_weights_after_loading` with a sticky async `CUDA_ERROR_ILLEGAL_ADDRESS`.
The surfacing point moves between adjacent lines of the nvfp4 marlin prep
(`gptq_marlin_repack`, `marlin_permute_scales`, `nvfp4_marlin_process_scales`)
after ~1-8 layers. Every crash wedges one or more GPUs at the RM level
(dmesg: `_scrubWaitAndSave` timeouts in the memory scrubber) while nvidia-smi
keeps looking healthy — the next run then fails `cudaSetDevice` on the wedged
GPU (cudaErrorDevicesUnavailable / OOM), which looks like a random worker race
but is not. Only a host reboot clears the wedge.

## What was ruled out (all single-process probes CLEAN)

| Probe | What it proves |
|---|---|
| `probe_marlin.py` | Full `prepare_nvfp4_moe_layer_for_marlin` with **real checkpoint tensors** (16 experts, any CUDA device incl. non-zero) — clean |
| `probe_repack_loop.py` | Cumulative `gptq_marlin_repack`: 10 layers x 576 calls in 0.4 s — clean |
| `probe_ballast.py` | Same with 38 GiB of ballast pre-allocated (engine-like memory pressure) — clean |
| `probe_multiproc.py` | 5 spawned procs + NCCL world + concurrent repack loops on all 5 GPUs — clean |

In-engine mitigations that did NOT help: `--safetensors-load-strategy eager`
(mmap/BTRFS theory), `NCCL_P2P_DISABLE=1`, serializing the marlin prep across
ranks with a flock, serializing the whole per-rank weight loading loop,
skipping the MLA absorb, running the scale pipeline on CPU (this one moved the
crash from the scale permute to the repack — kept anyway as defense).
`CUDA_LAUNCH_BLOCKING=1` pinned the fault inside the repack/scale kernels
themselves, i.e. the context is already poisoned when they run.

## Standing theory

PCIe link instability on the CMP 170HX's pinned Gen2 x4 links under 5-way
concurrent load-phase DMA: the host shows chronic corrected AER
Data-Link-Layer replay timeouts (`aer_layer=Data Link Layer, [12] Timeout`)
during loading, exactly on the GPU root ports. A retrain mid-DMA would
produce an async fault that surfaces at the next kernel — matching the moving
crash location and the wedge cascade. Unproven; the box's DeepSeek-V4 PP4
stack (FP8 marlin, different image) loads fine, so it is specific to this
workload's kernel mix or image build.

## Outcome / upgrade path

`--moe-backend humming` (Triton fp4, no CUDA repack kernels) loads and serves
cleanly: ~43 tok/s single-stream, 32k ctx, PP=5. If you need Marlin-grade
throughput, the deterministic fix is to **pre-repack offline**: run the
per-layer `prepare_nvfp4_moe_layer_for_marlin` single-process (proven clean,
fast), save the marlin-format tensors, and patch the engine to load them
instead of computing at startup — then only the (different) marlin GEMM
kernels run at serve time.
