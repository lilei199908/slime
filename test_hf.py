import torch
import torch.distributed as dist
from packaging import version
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoModelForCausalLM

dist.init_process_group(backend="nccl")
mesh = init_device_mesh("cpu", mesh_shape=((dist.get_world_size()),), mesh_dim_names=("dp",))
dp_mesh = mesh["dp"]
hf_path = "/cloud/oss_checkpoints/Qwen3/Qwen3-4B"


def apply_fsdp2(model, mesh=None, cpu_offload=False):
    """Apply FSDP v2 to the model.

    Args:
        model: The model to wrap with FSDP
        mesh: Optional DeviceMesh for FSDP. If None, uses all ranks.
        cpu_offload: If True, offload parameters, gradients, and optimizer states
            to CPU. The optimizer step will run on CPU. (Default: False)

    Ref: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py
    """
    # Import FSDP v2 components based on PyTorch version
    if version.parse(torch.__version__) >= version.parse("2.6"):
        from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard
    elif version.parse(torch.__version__) >= version.parse("2.4"):
        from torch.distributed._composable.fsdp import fully_shard
        from torch.distributed._composable.fsdp.fully_shard import CPUOffloadPolicy
    else:
        raise ImportError("FSDP v2 not available")

    offload_policy = CPUOffloadPolicy() if cpu_offload else None

    layer_cls_to_wrap = model._no_split_modules
    assert len(layer_cls_to_wrap) > 0 and layer_cls_to_wrap[0] is not None

    modules = [
        module
        for name, module in model.named_modules()
        if module.__class__.__name__ in layer_cls_to_wrap
        or (isinstance(module, torch.nn.Embedding) and not model.config.tie_word_embeddings)
    ]

    # Apply FSDP to each module (offload_policy=None is equivalent to not passing it)
    for module in modules:
        fully_shard(module, mesh=mesh, offload_policy=offload_policy)

    # Apply FSDP to the top-level model
    fully_shard(model, mesh=mesh, offload_policy=offload_policy)

    return model


model = AutoModelForCausalLM.from_pretrained(
    hf_path,
    trust_remote_code=True,
)
import psutil

print(psutil.virtual_memory())
print(psutil.cpu_percent(interval=1))

apply_fsdp2(model, mesh=mesh, cpu_offload=False)

print(psutil.virtual_memory())
print(psutil.cpu_percent(interval=1))
import time

time.sleep(1000)
