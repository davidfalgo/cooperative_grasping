# Description: Generic Trainer for CEModel (fit/evaluate/checkpoint/resume), plus a
# format_version-1 rich checkpoint and a load_model dispatcher covering both the new
# format and legacy flat state dicts (routes to CEModel.from_legacy_checkpoint).

import random

import numpy as np
import torch
from torch import nn

from ..config import CEModelConfig
from ..core import CEModel, build_model, parse_legacy_filename
from .metrics import hit_at_k, classification_metrics


class EarlyStopping:
    """Equivalent semantics to the legacy trainer.py EarlyStopping (patience,
    min_delta, restore_best_weights), reimplemented here so `general.training`
    never imports the legacy module (which imports wandb at module scope)."""

    def __init__(self, patience=10, min_delta=0.0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float("inf")
        self.best_state_dict = None
        self.wait = 0

    def step(self, current_loss, model):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
            self.wait = 0
            return False
        self.wait += 1
        if self.wait >= self.patience:
            if self.restore_best_weights and self.best_state_dict is not None:
                model.load_state_dict(self.best_state_dict)
            return True
        return False


def _to_device(inputs, device):
    return {k: v.to(device) for k, v in inputs.items()}


def _capture_rng_state():
    return {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class Trainer:
    def __init__(self, model, loss, optimizer, *, scheduler=None, device="cuda:0",
                 amp=False, grad_clip_norm=None, logger=None, early_stopping=None,
                 checkpoint_dir=None):
        self.model = model.to(device)
        self.loss_fn = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.amp = amp
        self.grad_clip_norm = grad_clip_norm
        self.logger = logger
        self.early_stopping = early_stopping
        self.checkpoint_dir = checkpoint_dir

        self._device_type = "cuda" if "cuda" in str(device) else "cpu"
        self.scaler = torch.amp.GradScaler(enabled=amp)
        self._epoch = 0
        self._global_step = 0

    def _run_batch(self, batch, train):
        (center_inputs, context_inputs), label, mask = batch
        center_inputs = _to_device(center_inputs, self.device)
        context_inputs = _to_device(context_inputs, self.device)
        label = label.to(self.device)
        mask = mask.to(self.device)

        with torch.autocast(device_type=self._device_type, enabled=self.amp):
            pred = self.model(center_inputs, context_inputs)
            loss = self.loss_fn(pred, label, mask)

        if train:
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            if self.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return pred.detach(), loss, label, mask

    def fit(self, train_loader, val_loader=None, epochs=80):
        history = {"train_loss": [], "train_step_loss": [], "val_loss": []}
        is_plateau = isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

        end_epoch = self._epoch + epochs
        while self._epoch < end_epoch:
            self.model.train()
            epoch_loss, total_pairs = 0.0, 0
            for batch in train_loader:
                _, loss, _, mask = self._run_batch(batch, train=True)

                valid_pairs = max(mask.sum().item(), 1)
                epoch_loss += loss.item() * valid_pairs
                total_pairs += valid_pairs
                history["train_step_loss"].append(loss.item())

                if self.logger is not None:
                    self.logger.log({"train/batch_loss": loss.item()}, step=self._global_step)
                self._global_step += 1

            history["train_loss"].append(epoch_loss / max(total_pairs, 1))

            val_loss = None
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                val_loss = val_metrics["loss"]
                history["val_loss"].append(val_loss)
                if self.logger is not None:
                    self.logger.log({f"val/{k}": v for k, v in val_metrics.items()
                                      if isinstance(v, (int, float))}, step=self._global_step)

            if self.scheduler is not None:
                if is_plateau:
                    self.scheduler.step(val_loss if val_loss is not None else history["train_loss"][-1])
                else:
                    self.scheduler.step()

            self._epoch += 1

            if val_loss is not None and self.early_stopping is not None:
                if self.early_stopping.step(val_loss, self.model):
                    break

        return history

    @torch.no_grad()
    def evaluate(self, loader, ks=(1, 3, 5)):
        self.model.eval()
        all_logits, all_labels, all_masks = [], [], []
        total_loss, total_pairs = 0.0, 0

        for batch in loader:
            pred, loss, label, mask = self._run_batch(batch, train=False)
            valid_pairs = max(mask.sum().item(), 1)
            total_loss += loss.item() * valid_pairs
            total_pairs += valid_pairs
            all_logits.append(pred.cpu())
            all_labels.append(label.cpu())
            all_masks.append(mask.cpu())

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)
        masks = torch.cat(all_masks)

        metrics = {"loss": total_loss / max(total_pairs, 1)}
        metrics["hit_at_k"] = hit_at_k(logits, labels, masks, ks=ks)
        metrics.update(classification_metrics(logits, labels, masks))
        return metrics

    def save_checkpoint(self, path, *, epoch=None, metrics=None):
        if self.model.config is None:
            raise ValueError(
                "model.config is None; build the model via build_model(config) "
                "(or set model.config manually) to enable checkpointing"
            )
        ckpt = {
            "format_version": 1,
            "model_config": self.model.config.to_dict(),
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler_state": self.scaler.state_dict() if self.amp else None,
            "epoch": self._epoch if epoch is None else epoch,
            "global_step": self._global_step,
            "best_metric": metrics,
            "rng": _capture_rng_state(),
        }
        torch.save(ckpt, path)

    @classmethod
    def resume(cls, path, *, optimizer_factory, scheduler_factory=None, loss,
               device="cuda:0", amp=False, grad_clip_norm=None, logger=None,
               early_stopping=None, checkpoint_dir=None, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        if ckpt.get("format_version") != 1:
            raise ValueError(f"Unsupported checkpoint format_version {ckpt.get('format_version')!r}")

        model = build_model(CEModelConfig.from_dict(ckpt["model_config"]))
        model.load_state_dict(ckpt["model_state"])

        optimizer = optimizer_factory(model.parameters())
        optimizer.load_state_dict(ckpt["optimizer_state"])

        scheduler = None
        if scheduler_factory is not None:
            scheduler = scheduler_factory(optimizer)
            if ckpt.get("scheduler_state") is not None:
                scheduler.load_state_dict(ckpt["scheduler_state"])

        trainer = cls(model, loss, optimizer, scheduler=scheduler, device=device, amp=amp,
                       grad_clip_norm=grad_clip_norm, logger=logger,
                       early_stopping=early_stopping, checkpoint_dir=checkpoint_dir)

        if ckpt.get("scaler_state") is not None:
            trainer.scaler.load_state_dict(ckpt["scaler_state"])

        trainer._epoch = ckpt.get("epoch", 0)
        trainer._global_step = ckpt.get("global_step", 0)

        if ckpt.get("rng") is not None:
            _restore_rng_state(ckpt["rng"])

        return trainer


def load_model(path, config=None, map_location="cpu"):
    obj = torch.load(path, map_location=map_location, weights_only=False)

    if isinstance(obj, dict) and "format_version" in obj:
        model = build_model(CEModelConfig.from_dict(obj["model_config"]))
        model.load_state_dict(obj["model_state"])
        return model

    if isinstance(obj, dict) and obj and all(k.startswith("0.") or k.startswith("1.") for k in obj):
        if config is not None:
            return CEModel.from_legacy_checkpoint(path, config, map_location=map_location)
        try:
            parsed = parse_legacy_filename(path)
        except ValueError as e:
            raise ValueError(
                f"'{path}' is a legacy flat state dict but its filename doesn't match the "
                "legacy naming pattern, so model_version/embedding_size can't be auto-derived. "
                "Pass config=CEModelConfig(...) explicitly."
            ) from e
        raise ValueError(
            f"'{path}' is a legacy checkpoint (model_version={parsed['model_version']!r}, "
            f"embedding_size={parsed['embedding_size']}); pass config=CEModelConfig(...) built "
            "with these values plus `feature_structure` (not recoverable from the filename or "
            "state dict) to load it."
        )

    raise ValueError(
        f"Unrecognized checkpoint format at '{path}': expected a dict with 'format_version' "
        "or a flat legacy state dict with '0.'/'1.' key prefixes."
    )
