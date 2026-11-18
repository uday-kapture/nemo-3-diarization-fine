"""LoRA on any nn.Module via PEFT's low-level API, merged back into plain nn.Linear for export."""
import re

import torch
from peft import LoraConfig, get_peft_model_state_dict, inject_adapter_in_model
from peft.tuners.lora import LoraLayer
from torch import nn


def add_lora(module: nn.Module, r: int = 16, alpha: int = 32, dropout: float = 0.05) -> list[str]:
    names = [n for n, m in module.named_modules() if isinstance(m, nn.Linear)]
    # Exact full paths: PEFT suffix-matches list entries, so a bare "0" would also hit e.g. a block at "layers.0".
    target = "|".join(map(re.escape, names))
    inject_adapter_in_model(LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=target), module)
    for n, p in module.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    return names


def merge_lora(module: nn.Module) -> None:
    for name, layer in [(n, m) for n, m in module.named_modules() if isinstance(m, LoraLayer)]:
        layer.merge()
        parent, _, child = name.rpartition(".")
        setattr(module.get_submodule(parent), child, layer.get_base_layer())
    # Stale once merged; left in place, a later add_lora would see an existing adapter config.
    module.__dict__.pop("peft_config", None)


def lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return get_peft_model_state_dict(module)
