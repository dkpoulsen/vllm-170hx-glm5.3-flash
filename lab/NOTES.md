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

## Wall 2 — full day of attribution (2026-08-29 evening)

Threshold found empirically: contexts <= ~55.5k WORK, >= ~61k CRASH (Xid 31
wild-pointer faults at MLA layers, GEMM victim; rank-agnostic — hit rank0
layer 3 and rank3 layer 27). NOT fixed by --enforce-eager or disabling
prefix caching (both still crashed) — cudagraphs and prefix cache are
exonerated. Shipped config: cap 49152, graphs+prefix ON, breakable
cudagraphs OFF, CUDA_LAUNCH_BLOCKING removed.

Components individually exonerated by standalone probes (probe_mla_ctx.py,
probe_mla_multiseq.py — all bit-exact vs references, no faults, sanitizer
not needed): gather_and_maybe_dequant_cache (incl. engine page size 4352,
fragmented pools, high page ids — int64-safe), kv_b_proj GEMM, custom
Triton MLA prefill kernel (context + causal modes, ragged multi-seq),
merge_attn_states, plan_mla_context_chunks + builder metadata (read clean;
seq_lens_cpu_upper_bound is exact for prefill rows per its docstring).

Key engine facts learned: attention block size padded to 4352 tokens to
match mamba state pools ("Add 10 padding layers" hybrid KV layout across
PP=5); fork defaults to 2048-token chunked-prefill budget; fork auto-enables
experimental breakable cudagraphs (VLLM_USE_BREAKABLE_CUDAGRAPH=1) and rank3
threw C++ exceptions during piecewise graph profiling — kept disabled.

Remaining suspects (untested): engine-side integration of the gather with
the hybrid 4352-page + padding-layer pool layout (e.g. per-layer block-id
offsets vs padding slots), the KDA/mamba state pool seam at chunk
boundaries, or a fork custom op beyond the gather. Next session: extend
probes to the mamba state-pool layout, or bisect in-engine by masking
context-chunk gathers (force small chunk budget via workspace override).

Crash signature reference (5 crashes today): Xid 31 GRAPHICS
FAULT_PDE/REGION_VIOLATION on the rank processing an MLA layer
(27/28@rank3 x3, 3@rank0 x1, 27@rank3 x1); DIAG surfaces at the MLA
layer's kv_b_proj cublasGemmEx or first op; one dual-GPU simultaneous
fault pair (GPU0 read + GPU3 write). Every crash during chunked prefill
of a long context; recovery = host reboot (GPU3 wedges at RM level,
set_device retries fail).

## Wall 2 — 8192 chunk-budget A/B (2026-08-29 night)

User-requested experiment: --max-num-batched-tokens 8192 (vs fork default
2048). RESULT: wall moved but persists — the 60k provoke (initial prefill
60.7k + turns) survived to 69.5k and crashed at EXACTLY
num_computed_tokens=69632 scheduling 9 tokens (VllmWorker-0 died, Xid 31
READ fault PDE on GPU0; engine dump_input captured the scheduler state).
So: 2048 budget -> wall ~60.7k; 8192 budget -> wall ~69.6k. The trigger
is absolute-position/chunk-phase dependent, NOT chunk-size dependent —
consistent with an indexing product (position x ~30k-class stride, or
position-dependent pool offset) wrapping int32 only for certain phases.
Reverted to the validated 49152 cap. Next root-cause idea: the engine's
own dump_input.py output shows num_common_prefix_blocks + per-group
new_block_ids — capture a few of these at crash to correlate block-id
magnitudes with the fault (block-id x 4352 x 512 x slots overflow?).

## Alternative stack: vowstar/vllm-sm80 (2026-08-30)

Tried the glm53-sm80 branch of github.com/vowstar/vllm-sm80 — an
independent sm_80 fork developed on identical hardware (5x CMP 170HX,
PP5, driver 610.43.x). Completely different serving shape from ours:
sparse MLA (DSA indexer, index_topk=2048 from the stock config) via a
Triton sm_80 kernel, fp8 e4m3 latent KV (software dequant in-kernel),
marlin W4A16 MoE, their own prefix-caching fixes. No overlays needed.

Build (64-core box, ~2 h, 1.4 TB free disk):
  cd /root/vllm-sm80 && podman build \
    --build-arg torch_cuda_arch_list="8.0" --build-arg max_jobs=40 \
    -t vllm-sm80:latest -f docker/Dockerfile .
(default max_jobs=2 is glacial; default arch list builds 7 variants)

Launch fixes vs the upstream README run command:
1. Image entrypoint already execs `vllm serve` — pass `/model ...` as
   CMD, NOT `vllm serve /model ...` (argparse rejects the duplicate).
2. `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` must be a multiple of OUR
   scheduler_block_size. Their README says 143360 (16 pages of 8960) but
   our run computes scheduler_block_size 8704 (layer partition/draft
   differences) -> use 139264 (16 x 8704). Error is explicit about it.
3. MTP x3 spec decode + `--moe-backend marlin` is INCOMPATIBLE with the
   mbehr90 checkpoint: layer-45 draft experts are stored UNQUANTIZED
   (plain `mlp.experts.N.*_proj.weight`, no nvfp4 packing) ->
   UnquantizedFusedMoEMethod rejects moe_backend='marlin' on PP4 at
   draft init. Either drop --speculative-config (what run-sm80.sh does;
   SPEC_MTP=1 re-enables) or switch --moe-backend triton. vowstar's own
   checkpoint presumably quantizes the draft.

Validated on our rig (all with dmesg Xid count 0 throughout):
- 95 s weight load per rank (vs 20 min serialized on the stock stack)
- KV pool 7,490,963 tokens (fp8) vs 3.9M (bf16) on the old stack
- provoke_sized: 76k, 131k, 262k, 524k, 1,039k-token prompts ALL CLEAN;
  1M cold prefill 205 s, cached appends 7 s/turn. WALL 2 DOES NOT EXIST
  in this stack — 1M context (the model max) serves.
- Single-stream decode ~44 tok/s idle (no MTP); ~16 tok/s while a 1M-
  context request decodes concurrently. glm47 reasoning parser DOES split
  thinking here — under a nonstandard "reasoning" key (not
  reasoning_content); content is clean either way.

Operational: `./run-sm80.sh` (stops nothing by itself — stop glm53-serve
first: `systemctl stop glm53-serve`). Switch back with
`podman rm -f glm53-sm80 && systemctl start glm53-serve`.
