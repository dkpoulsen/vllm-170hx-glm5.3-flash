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

## Wall 2 — second int32 overflow (open, 2026-08-29)

After the KDA patches (wall 1), a fresh engine run with max-model-len 1M
crashed ~90 s into a ~323k-token prefill request:

- Xid 31 on GPU4 (PCI 62:00) — the LAST PP rank (layers 36-44): MMU fault,
  GRAPHICS engine, REGION_VIOLATION VIRT_WRITE — same OOB-write signature
  as wall 1, different GPU/kernel.
- The kernel-level KDA probe (`lab/probe_fla.py`) is clean at 270k, so this
  site is in a different prefill-path kernel that only runs in-engine.
- Suspects (unverified): the MLA context-chunk gather
  (`ops.gather_and_maybe_dequant_cache` — paged-slot × 512 latent math),
  another absolute-position kernel on the last rank, or an mHC tilelang op.
- Method that worked for wall 1: kernel-level probe with
  CUDA_LAUNCH_BLOCKING=1 for exact attribution, then int64-widen the index
  products (and re-cast block-ptr shapes to int32 — Triton requires
  `offsets/block_shape` int32 while strides/bases may be int64).
- Working posture: `--max-model-len 262144` (below both walls; wall 1
  patched anyway). Each in-engine failure costs a host reboot (driver wedge).

## Wall 2 — refined (2026-08-29, second occurrence)

Crash recurred BELOW the 262144 cap: Xid 31 OOB write on rank 3 (GPU3,
layer 28) during a 256-token step of a ~250k-class context request that
arrived seconds after startup. With the earlier sighting on the last rank
(~323k prefill), wall 2 is position-triggered somewhere between ~128k and
~250k. Working hypothesis: a 16384-stride int32 overflow (2^31 / 16384 =
131072) — mHC-shaped or MLA-context-gather indexing. Safe cap lowered to
131072; oversized requests now fail with a clean 400 instead of wedging
the driver. Attribution data: the in-engine layer DIAG caught it at
layer 28; the faulting kernel itself is still unidentified — reproduce
with a single-proc decode/context-gather probe at >=150k before patching.
