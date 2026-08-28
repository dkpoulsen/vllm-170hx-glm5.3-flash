import time
import types
import glob
import torch
from safetensors.torch import safe_open

from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_nvfp4_moe_layer_for_marlin,
)

dev = "cuda:3"
E, K, N = 16, 4096, 2048
L = 10
base = "/models/GLM-5.3-Flash-nvfp4"

def load_expert_tensors(layer, count):
    out = {}
    need = count * 9
    for f in sorted(glob.glob(base + "/model-*.safetensors")):
        with safe_open(f, framework="pt") as sf:
            keys = set(sf.keys())
            for e in range(count):
                for p in ("gate_proj", "up_proj", "down_proj"):
                    for s in ("weight_packed", "weight_scale", "weight_global_scale"):
                        k = f"model.language_model.layers.{layer}.mlp.experts.{e}.{p}.{s}"
                        if k in keys and k not in out:
                            out[k] = sf.get_tensor(k)
        if len(out) == need:
            break
    assert len(out) == need, f"got {len(out)}/{need}"
    return out

t0 = time.time()
t = load_expert_tensors(L, E)
print(f"loaded {len(t)} tensors in {time.time()-t0:.1f}s")

def stack(p, s):
    ts = [t[f"model.language_model.layers.{L}.mlp.experts.{e}.{p}.{s}"] for e in range(E)]
    return torch.stack(ts).to(dev)

w13 = torch.cat([stack("gate_proj", "weight_packed"), stack("up_proj", "weight_packed")], dim=1).contiguous()
w2 = stack("down_proj", "weight_packed").contiguous()
s13 = torch.cat([stack("gate_proj", "weight_scale"), stack("up_proj", "weight_scale")], dim=1).contiguous()
s2 = stack("down_proj", "weight_scale").contiguous()
g13 = torch.cat([stack("gate_proj", "weight_global_scale"), stack("up_proj", "weight_global_scale")], dim=1).contiguous()
g2 = stack("down_proj", "weight_global_scale").contiguous()
print("shapes:", w13.shape, w2.shape, s13.shape, s2.shape, g13.shape, g2.shape)
print("scale stats: max", s13.float().max().item(), s2.float().max().item())

layer = types.SimpleNamespace(
    num_experts=E,
    hidden_size=K,
    intermediate_size_per_partition=N,
    params_dtype=torch.bfloat16,
    is_act_and_mul=True,
    w13_weight=w13,
    w2_weight=w2,
    w13_weight_scale=s13,
    w2_weight_scale=s2,
    w13_weight_global_scale=g13,
    w2_weight_global_scale=g2,
    # checkpoint has no input_global_scale (W4A16) — mirror the real
    # engine's torch.empty garbage exactly.
    w13_input_global_scale=torch.empty(E, 2, dtype=torch.float32, device=dev),
    w2_input_global_scale=torch.empty(E, dtype=torch.float32, device=dev),
    workspace=None,
)

t0 = time.time()
result = prepare_nvfp4_moe_layer_for_marlin(
    layer,
    w13=w13,
    w13_scale=s13,
    w13_scale_2=g13,
    w2=w2,
    w2_scale=s2,
    w2_scale_2=g2,
    is_act_and_mul=True,
)
torch.cuda.synchronize()
print(f"FULL MARLIN PREP CLEAN ({time.time()-t0:.1f}s)")
print("result:", [tuple(x.shape) if hasattr(x, 'shape') else x for x in result[:6]])
