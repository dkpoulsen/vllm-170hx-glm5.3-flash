import torch

from vllm.third_party.flash_linear_attention.ops.kda import chunk_kda_with_fused_gate

torch.manual_seed(0)
dev = "cuda:0"
H, K, V = 64, 128, 128
BT = 64  # FLA_CHUNK_SIZE


def run(total, cu_list, tag):
    cu = torch.tensor(cu_list, dtype=torch.int32, device=dev)
    n = len(cu_list) - 1
    q = torch.randn(1, total, H, K, dtype=torch.bfloat16, device=dev) * 0.05
    k = torch.randn(1, total, H, K, dtype=torch.bfloat16, device=dev) * 0.05
    v = torch.randn(1, total, H, V, dtype=torch.bfloat16, device=dev) * 0.05
    raw_g = -torch.rand(1, total, H, K, dtype=torch.bfloat16, device=dev) * 0.02
    beta = torch.rand(1, total, H, dtype=torch.bfloat16, device=dev) * 0.5 + 0.5
    A_log = torch.randn(H, dtype=torch.float32, device=dev).sigmoid().log()
    g_bias = torch.zeros(H, dtype=torch.float32, device=dev)
    st = torch.zeros(n, H, K, V, dtype=torch.float32, device=dev)
    out, final = chunk_kda_with_fused_gate(
        q=q,
        k=k,
        v=v,
        raw_g=raw_g,
        beta=beta,
        A_log=A_log,
        g_bias=g_bias,
        scale=K**-0.5,
        initial_state=st,
        output_final_state=True,
        cu_seqlens=cu,
    )
    torch.cuda.synchronize()
    print(
        f"{tag}: OK out={tuple(out.shape)} finite={bool(torch.isfinite(out.float()).all())} "
        f"absmax={out.float().abs().max().item():.4f}",
        flush=True,
    )
    return out, final


# Short multi-sequence: numerical regression vs stock
short = run(1500, [0, 700, 1500], "short patched")
torch.save({"out": short[0].cpu(), "final": short[1].cpu()}, "/work/probe-out/fla_short_patched.pt")
print("SHORT PASS", flush=True)

# Long single sequence past the 2^18 boundary (262144 tokens): int32 overflow
# territory for token*8192 element offsets.
long_total = 270000
out, final = run(long_total, [0, long_total], "long patched")
print("LONG PASS", flush=True)
