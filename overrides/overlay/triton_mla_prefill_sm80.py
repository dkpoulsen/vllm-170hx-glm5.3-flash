# SPDX-License-Identifier: Apache-2.0
# Triton ragged (varlen) MLA prefill backend for sm_80 — CMP 170HX patch.
#
# The glm53-flash image only ships ragged MLA prefill kernels for
# Hopper/Blackwell (FA3/FA4, FlashInfer, TRT-LLM) plus an FA2 path restricted
# to three MLA dimension combos. GLM-5.3-Flash is NoPE MLA
# (qk_nope=256, qk_rope=0, v=256) which none of them accept on sm_80, so this
# backend wraps a Triton varlen prefill kernel (ported from
# vllm/v1/attention/ops/triton_prefill_attention.py) extended to emit softmax
# LSE (flash_attn varlen layout: [nheads, total_q]) and to take separate q/k
# ragged offsets for context chunks.
#
# Head-dim 256 on GA100 (164 KB smem/SM) forces conservative tiles:
# BLOCK_M=128 x BLOCK_N=64 with num_stages=1 -> ~128 KB.

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.mla.prefill.base import (
    MLADimensions,
    MLAPrefillBackend,
)
from vllm.v1.attention.backends.mla.prefill.registry import (
    MLAPrefillBackendEnum,
    register_mla_prefill_backend,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )

logger = init_logger(__name__)

RCP_LN2 = 1.4426950408889634  # 1 / ln(2)
LN2 = 0.6931471805599453


@triton.jit
def _mla_prefill_kernel(
    Q,
    K,
    V,
    B_Start_Loc_Q,
    B_Start_Loc_K,
    B_Seqlen_Q,
    B_Seqlen_K,
    Out,
    Lse,
    sm_scale,
    lse_stride_h,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    kv_group_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    seq_len_q = tl.load(B_Seqlen_Q + cur_batch)
    seq_len_k = tl.load(B_Seqlen_K + cur_batch)
    start_q = tl.load(B_Start_Loc_Q + cur_batch)
    start_k = tl.load(B_Start_Loc_K + cur_batch)

    block_start_loc = BLOCK_M * start_m

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (
        (start_q + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    off_k = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None]
    off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :]

    mask_dk = offs_d < Lk
    mask_dv = offs_d < Lv

    q = tl.load(
        Q + off_q,
        mask=(offs_m[:, None] < seq_len_q) & (mask_dk[None, :]),
        other=0.0,
    )

    k_ptrs = K + off_k
    v_ptrs = V + off_v

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < seq_len_q, 1, 0)

    end_n = seq_len_k
    if IS_CAUSAL:
        end_n = tl.minimum(end_n, (start_m + 1) * BLOCK_M)
    end_n_limit = block_mask * end_n

    for start_n in range(0, end_n_limit, BLOCK_N):
        pos_q = offs_m[:, None]
        pos_k = start_n + offs_n[None, :]

        mask = pos_k < seq_len_k
        if IS_CAUSAL:
            mask &= pos_q >= pos_k

        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(
            k_ptrs + (start_k + start_n) * stride_kbs,
            mask=(pos_k < seq_len_k) & (mask_dk[:, None]),
            other=0.0,
        )

        qk = tl.dot(q, k)
        qk = tl.where(mask, qk * sm_scale, -1.0e8)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(
            v_ptrs + (start_k + start_n) * stride_vbs,
            mask=((start_n + offs_n[:, None]) < seq_len_k) & (mask_dv[None, :]),
            other=0.0,
        )
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        m_i = m_ij

    acc = acc / l_i[:, None]
    off_o = (
        (start_q + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :]
    )
    tl.store(
        Out + off_o,
        acc,
        mask=(offs_m[:, None] < seq_len_q) & (mask_dv[None, :]),
    )
    # Natural-log LSE (flash_attn_varlen softmax_lse semantics: [H, total_q]),
    # converted from the kernel's exp2 space.
    lse = (m_i + tl.math.log2(l_i)) * 0.6931471805599453
    tl.store(
        Lse + cur_head * lse_stride_h + start_q + offs_m,
        lse,
        mask=offs_m < seq_len_q,
    )


def _mla_prefill_launch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    max_seqlen_q: int,
    causal: bool,
    sm_scale: float,
):
    """q: [tq, H, Dq]; k: [tk, Hkv, Dq]; v: [tk, Hkv, Dv]; o: [tq, H, Dv];
    lse: [H, tq] fp32. cu_*: [B+1] ragged offsets (int32/64)."""
    batch = cu_q.shape[0] - 1
    if batch == 0 or q.shape[0] == 0:
        return
    cu_q = cu_q.to(torch.int32, copy=False).contiguous()
    cu_k = cu_k.to(torch.int32, copy=False).contiguous()
    seq_q = (cu_q[1:] - cu_q[:-1]).contiguous()
    seq_k = (cu_k[1:] - cu_k[:-1]).contiguous()
    heads = q.shape[1]
    kv_group_num = max(1, q.shape[1] // k.shape[1])

    lk = q.shape[-1]
    block_d = triton.next_power_of_2(max(lk, v.shape[-1]))

    # GA100 smem budget (164 KB) with head dim 256: 128x64 tiles, 1 stage.
    block_m, block_n, num_warps = 128, 64, 8

    grid = (batch, heads, triton.cdiv(max(max_seqlen_q, 1), block_m))
    _mla_prefill_kernel[grid](
        q,
        k,
        v,
        cu_q,
        cu_k,
        seq_q,
        seq_k,
        o,
        lse,
        sm_scale * RCP_LN2,
        q.shape[0],
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        o.stride(0),
        o.stride(1),
        kv_group_num=kv_group_num,
        BLOCK_M=block_m,
        BLOCK_DMODEL=block_d,
        BLOCK_N=block_n,
        IS_CAUSAL=causal,
        Lk=lk,
        Lv=v.shape[-1],
        num_warps=num_warps,
        num_stages=1,
    )


class TritonMlaPrefillSm80(MLAPrefillBackend):
    """Ragged Triton MLA prefill for pre-Hopper GPUs (any MLA dims)."""

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_PREFILL_SM80"

    @classmethod
    def supports_mla_dimensions(cls, mla_dimensions: MLADimensions) -> bool:
        # Varlen Triton kernel is dimension-agnostic (MHA layout, GQA ok).
        return True

    @classmethod
    def supports_compute_capability(cls, device_capability) -> bool:
        return device_capability.major <= 9

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        md = self._prefill_metadata
        cu = md.query_start_loc
        batch = cu.shape[0] - 1
        o = (
            out
            if out is not None
            else torch.empty(
                q.shape[0], q.shape[1], v.shape[-1], dtype=q.dtype, device=q.device
            )
        )
        lse = torch.empty(
            q.shape[1], q.shape[0], dtype=torch.float32, device=q.device
        )
        _mla_prefill_launch(
            q,
            k,
            v,
            o,
            lse,
            cu_q=cu,
            cu_k=cu,
            max_seqlen_q=md.max_query_len,
            causal=True,
            sm_scale=self.scale,
        )
        if return_softmax_lse:
            return o, lse
        return o

    def run_prefill_context_chunk(
        self,
        chunk: "MLACommonPrefillMetadata.ContextChunk",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cu_q = chunk.query_start_loc
        o = (
            out
            if out is not None
            else torch.empty(
                q.shape[0], q.shape[1], v.shape[-1], dtype=q.dtype, device=q.device
            )
        )
        lse = torch.empty(
            q.shape[1], q.shape[0], dtype=torch.float32, device=q.device
        )
        _mla_prefill_launch(
            q,
            k,
            v,
            o,
            lse,
            cu_q=cu_q,
            cu_k=chunk.cu_seq_lens,
            max_seqlen_q=chunk.max_query_len,
            causal=False,  # context is unmasked
            sm_scale=self.scale,
        )
        return o, lse


register_mla_prefill_backend(
    MLAPrefillBackendEnum.CUSTOM,
    f"{TritonMlaPrefillSm80.__module__}.{TritonMlaPrefillSm80.__qualname__}",
)
logger.info_once("Registered TritonMlaPrefillSm80 as MLA prefill backend CUSTOM")
