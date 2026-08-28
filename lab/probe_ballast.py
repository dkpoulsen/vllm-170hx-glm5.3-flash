import time
import torch
from vllm import _custom_ops as ops

dev = "cuda:0"
E, K, N = 288, 4096, 2048
LAYERS = 10

# Fill the GPU with ~38 GiB of ballast to mirror the engine's
# post-load memory state before running the repack loop.
ballast = []
alloc = 0
target = 38 * 1024**3
while alloc < target:
    chunk = torch.empty(min(2 * 1024**3, target - alloc), dtype=torch.uint8, device=dev)
    chunk.fill_(0xAB)
    ballast.append(chunk)
    alloc += chunk.numel()
print(f"ballast: {alloc/1024**3:.1f} GiB", flush=True)

w13 = torch.randint(0, 255, (E, 2 * N, K // 2), dtype=torch.uint8, device=dev)
w2 = torch.randint(0, 255, (E, K, N // 2), dtype=torch.uint8, device=dev)
perm = torch.empty(0, dtype=torch.int, device=dev)

def repack(weight, size_n, size_k):
    out = None
    for i in range(weight.shape[0]):
        qweight = weight[i].view(torch.int32).T.contiguous()
        mq = ops.gptq_marlin_repack(
            b_q_weight=qweight, perm=perm, size_k=size_k, size_n=size_n,
            num_bits=4, is_a_8bit=False,
        )
        if out is None:
            out = torch.empty((weight.shape[0], *mq.shape), dtype=mq.dtype, device=mq.device)
        out[i] = mq
    return out

t0 = time.time()
for L in range(LAYERS):
    m13 = repack(w13, 2 * N, K)
    m2 = repack(w2, K, N)
    torch.cuda.synchronize()
    print(f"layer {L}: repack OK ({time.time()-t0:.1f}s)", flush=True)
print("BALLAST REPACK CLEAN")
