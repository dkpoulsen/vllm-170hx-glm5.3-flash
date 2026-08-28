import os
import torch
import torch.distributed as dist
import torch.multiprocessing as tmp


def worker(local_rank, world, init_file):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(world))
    torch.cuda.set_device(local_rank)
    dev = f"cuda:{local_rank}"
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        world_size=world,
        rank=local_rank,
    )
    dist.barrier()
    print(f"rank {local_rank}: NCCL up", flush=True)

    from vllm import _custom_ops as ops

    E, K, N = 288, 4096, 2048
    LAYERS = 3
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
                out = torch.empty(
                    (weight.shape[0], *mq.shape), dtype=mq.dtype, device=mq.device
                )
            out[i] = mq
        return out

    for L in range(LAYERS):
        m13 = repack(w13, 2 * N, K)
        m2 = repack(w2, K, N)
        torch.cuda.synchronize()
        print(f"rank {local_rank} layer {L}: repack OK", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    print(f"rank {local_rank}: DONE CLEAN", flush=True)


if __name__ == "__main__":
    tmp.start_processes(
        worker,
        args=(5, "/tmp/dist_init_probe"),
        nprocs=5,
        start_method="spawn",
    )
