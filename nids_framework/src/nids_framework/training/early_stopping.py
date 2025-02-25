import logging
import os
from typing import Optional

import torch
import numpy as np
from torch import nn


class EarlyStopping:

    __slots__ = [
        "best_score",
        "early_stop",
        "val_loss_min",
        "_patience",
        "_delta",
        "_counter",
        "best_model_wts"
    ]

    def __init__(self, patience: int = 7, delta: float = 0.0):
        self.best_score: Optional[float] = None
        self.early_stop: bool = False
        self.val_loss_min: float = np.inf
        self._patience: int = patience
        self._delta: float = delta
        self._counter: int = 0
        self.best_model_wts = None

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_loss, model)
            self.best_model_wts = model.state_dict()
        elif score < self.best_score * (1 - self._delta):
            self._counter += 1
            logging.info(f"EarlyStopping counter: {self._counter}/{self._patience}")
            if self._counter >= self._patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(val_loss, model)
            self.best_model_wts = model.state_dict()
            self._counter = 0

    def _save_checkpoint(self, val_loss: float, model: nn.Module, f_path: str = "checkpoints/checkpoint.pt") -> None:
        logging.info(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")
        os.makedirs(os.path.dirname(f_path), exist_ok=True)
        torch.save(model.state_dict(), f_path)
        self.val_loss_min = val_loss

    def restore_best_weights(self, model: nn.Module) -> None:
        if self.best_model_wts is not None:
            model.load_state_dict(self.best_model_wts)
        else:
            logging.warning("Best model weights not found. No restoration performed.")
