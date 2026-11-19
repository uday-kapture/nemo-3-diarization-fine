"""LoRA fine-tune of Nemotron 3 Diarization on NeMo's own loader and losses, exported as a merged .nemo."""
import argparse
import math
import time
from pathlib import Path

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, LambdaCallback, ModelCheckpoint
from nemo.collections.asr.models import SortformerEncLabelModel
from omegaconf import open_dict

from nemo_3_diarization import notify, tracking
from nemo_3_diarization.lora import add_lora, lora_state_dict, merge_lora

MERGED = "nemotron3-diar-lora-merged.nemo"


class ChatProgress(Callback):
    """Google Chat: start, each validation (val_loss against the best so far, ETA), a non-finite loss."""

    def __init__(self, chat: notify.GChat, name: str, batch_size: int):
        self.chat, self.name, self.batch_size = chat, name, batch_size
        self.loss = self.best = None
        self.bad = False

    def on_train_start(self, trainer, pl_module):
        self.t0 = time.perf_counter()
        trainable = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU"
        self.chat.send(notify.started_text(self.name, trainer.max_steps, self.batch_size, trainable, gpu))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.loss = float(outputs["loss"])
        if not math.isfinite(self.loss) and not self.bad:  # once per bad stretch, not every step
            self.chat.send(notify.bad_loss_text(self.name, trainer.global_step, self.loss))
        self.bad = not math.isfinite(self.loss)

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or "val_loss" not in trainer.callback_metrics:
            return
        v = float(trainer.callback_metrics["val_loss"])
        self.chat.send(notify.val_text(self.name, trainer.global_step, trainer.max_steps, v, self.best, self.loss,
                                       time.perf_counter() - self.t0))
        self.best = v if self.best is None or v < self.best else self.best


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
    ap.add_argument("--name", help="run name in MLflow and the Chat thread (default: <out-dir name>-<date-time>)")
    ap.add_argument("--mlflow-experiment", default="Nemotron-diarization-fine-tune")
    a = ap.parse_args()
    out = a.out_dir.resolve()
    name = a.name or f"{out.name}-{time.strftime('%Y%m%d-%H%M')}"
    try:
        logger = tracking.lightning_logger(a.mlflow_experiment, name, tags={"stage": "train"})
    except tracking.TrackingError as e:
        ap.error(str(e))
    out.mkdir(parents=True, exist_ok=True)
    (out / tracking.RUN_ID_FILE).write_text(logger.run_id + "\n")  # creates the run now: a bad server fails here
    chat = notify.GChat.from_env(thread=name)

    ckpt = ModelCheckpoint(dirpath=out / "ckpt", monitor="val_loss", save_top_k=1, verbose=True)
    # NeMo logs the loss to the logger only; print it so a run log shows it falling.
    show_loss = LambdaCallback(
        on_train_batch_end=lambda tr, pl, o, b, i: print(f"step {tr.global_step} loss {o['loss'].item():.4f}")
    )
    trainer = Trainer(
        devices=1, accelerator=a.accelerator, precision=a.precision, max_steps=a.max_steps,
        val_check_interval=a.val_every, check_val_every_n_epoch=None,
        callbacks=[ckpt, show_loss, ChatProgress(chat, name, a.batch_size)], logger=logger,
        default_root_dir=out,  # Lightning's own files land here, not in the cwd
    )

    model = SortformerEncLabelModel.from_pretrained("nvidia/Nemotron-3-Diarization")
    model.set_trainer(trainer)
    model.requires_grad_(False)  # not model.freeze(): NeMo's freeze() also calls eval(), disabling dropout/augmentation
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
    model.cfg = model.cfg  # NeMo captured the config MLflow gets at load; the setter refreshes it with the overrides
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f} %)")
    # "cli." keeps these apart from the model config NeMo/Lightning may log: MLflow refuses a changed param value
    logger.log_hyperparams({f"cli.{k}": str(v) for k, v in vars(a).items()} | {"trainable_params": trainable})

    t0 = time.perf_counter()
    try:
        # finally: export even on a crash or Ctrl-C (Lightning turns KeyboardInterrupt into sys.exit(1)) or with no
        # validation
        try:
            trainer.fit(model)
        finally:
            print(f"best checkpoint: {ckpt.best_model_path or 'NONE - exporting last weights'} (val_loss {ckpt.best_model_score})")
            if ckpt.best_model_path:
                model.load_state_dict(torch.load(ckpt.best_model_path, map_location="cpu", weights_only=False)["state_dict"])
            torch.save(lora_state_dict(model.encoder), out / "lora_adapter.pt")
            merge_lora(model.encoder)
            model.save_to(str(out / MERGED))
    except BaseException as e:  # incl. Ctrl-C; an export failure (e.g. a full disk) is reported too
        chat.send(notify.failed_text(name, e, trainer.global_step))
        raise
    best = ckpt.best_model_score
    chat.send(notify.finished_text(name, trainer.global_step, None if best is None else float(best),
                                   time.perf_counter() - t0, str(out / MERGED)))


if __name__ == "__main__":
    main()
