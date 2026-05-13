import torch
from collections import OrderedDict
from typing import Sequence, Dict

def stack_params(models: Sequence[torch.nn.Module]) -> Dict[str, torch.Tensor]:
    """
    Stack parameters from homogeneous models into a batched PyTree.
    """
    if len(models) == 0:
        raise ValueError("Empty model list")

    param_names = list(dict(models[0].named_parameters()).keys())

    stacked = OrderedDict()
    for name in param_names:
        tensors = [dict(m.named_parameters())[name] for m in models]
        stacked[name] = torch.stack(tensors, dim=0)

    return stacked


def stack_buffers(models: Sequence[torch.nn.Module]) -> Dict[str, torch.Tensor]:
    """
    Stack buffers from homogeneous models into a batched PyTree.
    """
    if len(models) == 0:
        raise ValueError("Empty model list")

    buffer_names = list(dict(models[0].named_buffers()).keys())

    stacked = OrderedDict()
    for name in buffer_names:
        tensors = [dict(m.named_buffers())[name] for m in models]
        stacked[name] = torch.stack(tensors, dim=0)

    return stacked


def stack_state(models: Sequence[torch.nn.Module]):
    """
    Returns (params, buffers) suitable for torch.func.functional_call.
    """
    return stack_params(models), stack_buffers(models)


def unstack_state(stacked_params: Dict[str, torch.Tensor], stacked_buffers: Dict[str, torch.Tensor]) -> Sequence[Dict[str, torch.Tensor]]:
    """
    Unstack batched PyTree into a list of PyTrees for each model.
    """
    n_models = next(iter(stacked_params.values())).shape[0]
    unstacked = []
    for i in range(n_models):
        params_i = {name: tensor[i] for name, tensor in stacked_params.items()}
        buffers_i = {name: tensor[i] for name, tensor in stacked_buffers.items()}
        unstacked.append(params_i | buffers_i)
    return unstacked