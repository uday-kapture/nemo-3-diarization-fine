import torch, torch.nn as nn
from nemo_3_diarization.lora import add_lora, merge_lora, lora_state_dict

def toy():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

def test_only_lora_trainable():
    m = toy(); add_lora(m)
    assert all("lora_" in n for n, p in m.named_parameters() if p.requires_grad)
    assert any(p.requires_grad for p in m.parameters())

def test_merge_preserves_output_and_removes_lora():
    m = toy(); add_lora(m)
    for n, p in m.named_parameters():
        if "lora_B" in n: nn.init.normal_(p)
    m.eval()  # LoRA dropout is active in train mode; merged weights reproduce the eval-mode output
    x = torch.randn(5, 4); y = m(x)
    assert len(lora_state_dict(m)) > 0
    merge_lora(m)
    assert torch.allclose(m(x), y, atol=1e-5)
    assert not any("lora" in k for k in m.state_dict())
    assert all(type(l) in (nn.Linear, nn.ReLU) for l in m)
