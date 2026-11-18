"""LoRA fine-tune of Nemotron 3 Diarization on NeMo's own loader and losses, exported as a merged .nemo."""
import argparse
from pathlib import Path

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LambdaCallback, ModelCheckpoint
from nemo.collections.asr.models import SortformerEncLabelModel
from omegaconf import open_dict

from nemo_3_diarization.lora import add_lora, lora_state_dict, merge_lora


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-manifest", type=Path, required=True)
    ap.add_argument("--val-manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("out"))
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--accelerator", default="gpu")
    ap.add_argument("--precision", default="bf16-mixed")
    a = ap.parse_args()
    out = a.out_dir.resolve()

    ckpt = ModelCheckpoint(dirpath=out / "ckpt", monitor="val_loss", save_top_k=1, verbose=True)
    # NeMo logs the loss to the logger only; print it so a run log shows it falling.
    show_loss = LambdaCallback(
        on_train_batch_end=lambda tr, pl, o, b, i: print(f"step {tr.global_step} loss {o['loss'].item():.4f}")
    )
    trainer = Trainer(
        devices=1, accelerator=a.accelerator, precision=a.precision, max_steps=a.max_steps,
        val_check_interval=a.val_every, check_val_every_n_epoch=None, callbacks=[ckpt, show_loss],
        default_root_dir=out,  # Lightning logs land here, not in the cwd
    )

    model = SortformerEncLabelModel.from_pretrained("nvidia/Nemotron-3-Diarization")
    model.set_trainer(trainer)
    model.requires_grad_(False)
    targets = add_lora(model.encoder, a.rank, a.alpha)
    print(f"LoRA targets ({len(targets)}):\n  " + "\n  ".join(targets))

    with open_dict(model.cfg):
        for ds, path in (("train_ds", a.train_manifest), ("validation_ds", a.val_manifest)):
            model.cfg[ds].manifest_filepath = str(path.resolve())
            model.cfg[ds].batch_size = a.batch_size
            model.cfg[ds].num_workers = a.num_workers
    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)

    model.setup_optimization({
        "name": "adamw", "lr": a.lr, "betas": [0.9, 0.98], "weight_decay": 0.001,
        "sched": {"name": "InverseSquareRootAnnealing", "warmup_steps": 200, "min_lr": 1e-6},
    })
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f} %)")

    trainer.fit(model)

    print(f"best checkpoint: {ckpt.best_model_path} (val_loss {ckpt.best_model_score})")
    model.load_state_dict(torch.load(ckpt.best_model_path, map_location="cpu", weights_only=False)["state_dict"])
    torch.save(lora_state_dict(model.encoder), out / "lora_adapter.pt")
    merge_lora(model.encoder)
    model.save_to(str(out / "nemotron3-diar-lora-merged.nemo"))


if __name__ == "__main__":
    main()
