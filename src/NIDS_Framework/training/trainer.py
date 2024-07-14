from typing import Tuple, Optional, Callable
import logging
from tqdm import tqdm
import functools

import numpy as np
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from training import metrics
from tools.utilities import trace_stats


class EarlyStopping:

    __slots__ = [
        "patience",
        "delta",
        "counter",
        "best_score",
        "early_stop",
        "val_loss_min",
    ]

    def __init__(self, patience: Optional[int] = 7, delta: Optional[float] = 0):
        self.patience: int = patience
        self.delta: float = delta
        self.counter: int = 0
        self.best_score: float = None
        self.early_stop: bool = False
        self.val_loss_min: float = np.inf

    def __call__(self, val_loss: float, model: nn.Module):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.best_score * self.delta:
            self.counter += 1
            logging.info(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module):
        logging.info(
            f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ..."
        )
        torch.save(model.state_dict(), "checkpoints/checkpoint.pt")
        self.val_loss_min = val_loss


class Trainer:

    __slots__ = [
        "_model",
        "_criterion",
        "_optimizer",
    ]

    def __init__(
        self, model: nn.Module, criterion: Optional[_Loss] = None, optimizer: Optional[Optimizer] = None 
    ) -> None:
        self._model: nn.Module = model
        self._criterion: _Loss = criterion
        self._optimizer: Optimizer = optimizer

    @trace_stats()
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

        if patience is not None and delta is not None:
            early_stopping = EarlyStopping(patience, delta)

        train_loss = 0.0
        self._model.train()

        for epoch in range(n_epoch):
            epoch_loss = self.train_one_epoch(train_data_loader, epoch_steps)
            train_loss += epoch_loss
            logging.info(f"Epoch: {epoch} Loss: {epoch_loss:.6f}.\n")

            if valid_data_loader:
                if (epoch + 1) % epochs_until_validation == 0:
                    validation_loss = self.validate(valid_data_loader)
                    early_stopping(validation_loss, self._model)
                    if early_stopping.early_stop:
                        logging.info(f"Early stopping in epoch: {epoch+1}")
                        self._model.load_state_dict(
                            torch.load("checkpoints/checkpoint.pt")
                        )
                        break
                    else:
                        self._model.train()

        train_loss /= epoch + 1
        logging.info("Done with training.")
        logging.info(f"Trained for {epoch + 1} epochs with loss: {train_loss:.6f}.\n")
        return train_loss

    def train_one_epoch(
        self, data_loader: DataLoader, epoch_steps: Optional[int] = None
    ) -> float:
        epoch_loss = 0.0

        if epoch_steps is None:
            for batch in tqdm(data_loader, desc="Training"):
                epoch_loss += self._train_one_batch(batch)
            epoch_loss /= len(data_loader)
        else:
            if epoch_steps > len(data_loader):
                raise ValueError(
                    f"Epoch steps must be less or at least equal to {len(data_loader)}."
                )
            epoch_iter = iter(data_loader)
            for _ in tqdm(range(epoch_steps), desc="Training"):
                batch = next(epoch_iter)
                epoch_loss += self._train_one_batch(batch)
            epoch_loss /= epoch_steps

        return epoch_loss

    def _train_one_batch(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        inputs, labels = batch

        self._optimizer.zero_grad()
        outputs = self._model(inputs)
        loss = self._criterion(outputs, labels)
        loss.backward()
        self._optimizer.step()

        return loss.item()

    def validate(self, data_loader: DataLoader) -> float:
        logging.info("Starting validation loop...")
        validation_loss = 0.0
        
        self._model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(data_loader, desc="Validating"):
                outputs = self._model(inputs)
                loss = self._criterion(outputs, labels)
                validation_loss += loss.item()

        validation_loss /= len(data_loader)

        logging.info("Done with validation.")
        logging.info(f"Validation loss: {validation_loss:.6f}.\n")
        return validation_loss

    @trace_stats()
    def test(self, data_loader: DataLoader, metric: metrics.Metric) -> None:
        if metric is None:
            raise ValueError("Please provide metic before testing.")
        logging.info(f"Starting test loop...")

        self._model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(data_loader, desc="Testing"):
                outputs = self._model(inputs)
                metric.step(outputs, labels)

        metric.apply()
        logging.info("Done with testing.")
        logging.info(f"{metric}\n")
        metric.save()

    def save_model(self, path: Optional[str] = "saves/model.pt") -> None:
        logging.info("Saving model weights...")
        torch.save(self._model.state_dict(), path)
        logging.info("Done")

    def load_model(self, path: Optional[str] = "saves/model.pt") -> None:
        logging.info("Loading model weights...")
        self._model.load_state_dict(torch.load(path))
        logging.info("Done")
