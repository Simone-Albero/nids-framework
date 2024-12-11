import logging
import os
from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from .metrics import Metric
from ..tools.utilities import trace_stats


class EarlyStopping:

    __slots__ = [
        "best_score",
        "early_stop",
        "val_loss_min",
        "_patience",
        "_delta",
        "_counter",
    ]

    def __init__(self, patience: int = 7, delta: float = 0):
        self.best_score: Optional[float] = None
        self.early_stop: bool = False
        self.val_loss_min: float = np.inf
        self._patience = patience
        self._delta = delta
        self._counter: int = 0

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_loss, model)
        elif score < self.best_score * (1 - self._delta):
            self._counter += 1
            logging.info(f"EarlyStopping counter: {self._counter}/{self._patience}")
            if self._counter >= self._patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(val_loss, model)
            self._counter = 0

    def _save_checkpoint(
        self,
        val_loss: float,
        model: nn.Module,
        f_path: str = "checkpoints/checkpoint.pt",
    ) -> None:
        logging.info(
            f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ..."
        )

        os.makedirs(os.path.dirname(f_path), exist_ok=True)
        torch.save(model.state_dict(), f_path)
        self.val_loss_min = val_loss


class Trainer:

    __slots__ = [
        "model",
        "criterion",
        "optimizer",
    ]

    def __init__(
        self,
        model: nn.Module,
        criterion: Optional[_Loss] = None,
        optimizer: Optional[Optimizer] = None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    #@trace_stats()
    def train(
        self,
        n_epoch: int,
        train_data_loader: DataLoader,
        epoch_steps: Optional[int] = None,
        epochs_until_validation: Optional[int] = None,
        valid_data_loader: Optional[DataLoader] = None,
        patience: Optional[int] = None,
        delta: Optional[float] = None,
    ) -> float:
        logging.info(f"Starting {n_epoch}-epoch training loop...")

        early_stopping = EarlyStopping(patience, delta) if patience and delta else None
        train_loss = 0.0

        for epoch in range(n_epoch):
            epoch_loss = self.train_one_epoch(train_data_loader, epoch_steps)
            train_loss += epoch_loss
            logging.info(f"Epoch {epoch+1}/{n_epoch} Loss: {epoch_loss:.6f}")

            if valid_data_loader and epochs_until_validation and (epoch + 1) % epochs_until_validation == 0:
                validation_loss = self.validate(valid_data_loader)
                if early_stopping:
                    early_stopping(validation_loss, self.model)
                    if early_stopping.early_stop:
                        logging.info(f"Early stopping at epoch {epoch+1}")
                        self.model.load_state_dict(torch.load("checkpoints/checkpoint.pt"))
                        break

        train_loss /= n_epoch
        logging.info(f"Training completed: {n_epoch} epochs, Average Loss: {train_loss:.6f}")
        return train_loss
    
    def train_one_epoch(
        self, data_loader: DataLoader, epoch_steps: Optional[int] = None
    ) -> float:
        epoch_loss = 0.0
        total_steps = min(epoch_steps, len(data_loader)) if epoch_steps else len(data_loader)
        data_iter = iter(data_loader)

        self.model.train()
        for _ in tqdm(range(total_steps), desc="Training"):
            epoch_loss += self._train_one_batch(next(data_iter))
            
        return epoch_loss / total_steps

    def _train_one_batch(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        inputs, labels = batch

        self.optimizer.zero_grad()
        outputs = self.model(inputs)

        loss = self.criterion(outputs, labels)
        # loss = self.criterion(outputs)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validate(self, data_loader: DataLoader) -> float:
        logging.info("Starting validation loop...")
        validation_loss = 0.0

        self.model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(data_loader, desc="Validating"):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                validation_loss += loss.item()

        validation_loss /= len(data_loader)

        logging.info("Done with validation.")
        logging.info(f"Validation loss: {validation_loss:.6f}.\n")
        return validation_loss

    #@trace_stats()
    def test(self, data_loader: DataLoader, metric: Optional[Metric] = None) -> None:
        logging.info(f"Starting test loop...")
        test_loss = 0.0

        self.model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(data_loader, desc="Testing"):
                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                if metric is not None:
                    metric.step(outputs, labels)

        test_loss /= len(data_loader)
        logging.info("Done with testing.")
        logging.info(f"Test loss: {test_loss:.6f}.\n")

        if metric is not None:
            metric.compute_metrics()
            logging.info(f"{metric}\n")
            metric.save()
