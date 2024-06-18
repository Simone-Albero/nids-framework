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

from tools import hook_system
from training import metrics
from utilities import trace_stats

class EarlyStopping:

    __slots__ = [
        "patience",
        "delta",
        "counter",
        "best_score",
        "early_stop",
        "val_loss_min",
    ]

    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
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

    def save_checkpoint(self, val_loss, model):
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
        "_device",
        "_hook_system",
    ]

    BEFORE_TRAIN = "before_train"
    BEFORE_EPOCH = "before_epoch"
    BEFORE_BATCH = "before_batch"
    BEFORE_VALIDATION = "before_validation"
    BEFORE_TEST = "before_test"

    AFTER_TRAIN = "after_train"
    AFTER_EPOCH = "after_epoch"
    AFTER_BATCH = "after_batch"
    AFTER_VALIDATION = "after_validation"
    AFTER_TEST = "after_test"

    def __init__(
        self, model: nn.Module, criterion: _Loss, optimizer: Optimizer
    ) -> None:
        self._model: nn.Module = model
        self._criterion: _Loss = criterion
        self._optimizer: Optimizer = optimizer
        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._hook_system = hook_system.HookSystem()

    def add_callback(self, event: str) -> Callable:
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            self._hook_system.register_hook(event, wrapper)
            return wrapper

        return decorator

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
        self._hook_system.execute_hooks(self.BEFORE_TRAIN)

        if patience is not None and delta is not None:
            early_stopping = EarlyStopping(patience, delta)

        train_loss = 0.0
        self._model.train()

        for epoch in range(n_epoch):
            epoch_loss = self.train_one_epoch(train_data_loader, epoch_steps)
            train_loss += epoch_loss
            logging.info(f"Epoch: {epoch} Loss: {epoch_loss:.6f}.\n")

            if valid_data_loader and epochs_until_validation:
                if (epoch + 1) % epochs_until_validation == 0 and (
                    epoch + 1
                ) != n_epoch:
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

        train_loss /= (epoch +1)
        logging.info("Done with training.")
        logging.info(f"Trained for {epoch + 1} epochs with loss: {train_loss:.6f}.\n")
        self._hook_system.execute_hooks(self.AFTER_TRAIN)
        return train_loss

    def train_one_epoch(
        self, data_loader: DataLoader, epoch_steps: Optional[int]
    ) -> float:
        self._hook_system.execute_hooks(self.BEFORE_EPOCH)
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

        self._hook_system.execute_hooks(self.AFTER_EPOCH)
        return epoch_loss

    def _train_one_batch(self, batch: Tuple) -> float:
        self._hook_system.execute_hooks(self.BEFORE_BATCH)
        inputs, labels = batch
        inputs = inputs.to(self._device)
        labels = labels.to(self._device)

        self._optimizer.zero_grad()
        outputs = self._model(inputs)
        loss = self._criterion(outputs, labels)

        #l1_norm = sum(param.abs().sum() for param in self._model.parameters())
        #loss += 0.01 * l1_norm
        loss.backward()
        self._optimizer.step()

        self._hook_system.execute_hooks(self.AFTER_BATCH)
        return loss.item()

    def validate(self, data_loader: DataLoader) -> float:
        logging.info("Starting validation loop...")
        self._hook_system.execute_hooks(self.BEFORE_VALIDATION)
        validation_loss = 0.0
        self._model.eval()

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Validating"):
                validation_loss += self._validate_one_batch(batch)

        validation_loss /= len(data_loader)

        logging.info("Done with validation.")
        logging.info(f"Validation loss: {validation_loss:.6f}.\n")
        self._hook_system.execute_hooks(self.AFTER_VALIDATION)
        return validation_loss

    def _validate_one_batch(self, batch: Tuple) -> float:
        inputs, labels = batch
        inputs = inputs.to(self._device)
        labels = labels.to(self._device)

        outputs = self._model(inputs)
        loss = self._criterion(outputs, labels)
        return loss.item()

    def test(
        self, data_loader: DataLoader, metric: metrics.Metric
    ) -> None:
        if metric is None:
            raise ValueError("Please provide metic before testing.")
        logging.info(f"Starting test loop...")
        self._hook_system.execute_hooks(self.BEFORE_TEST)
        self._model.eval()

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Testing"):
                outputs, labels = self._test_one_batch(batch)
                metric.step(outputs, labels)

        metric.apply()
        logging.info("Done with testing.")
        logging.info(f"{metric}\n")
        self._hook_system.execute_hooks(self.AFTER_TEST)

    def _test_one_batch(
        self, batch: Tuple,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, labels = batch
        inputs = inputs.to(self._device)
        labels = labels.to(self._device)

        outputs = self._model(inputs)
        return outputs, labels

    def save_model(self, path: Optional[str] = "saves/model.pt") -> None:
        logging.info("Saving model weights...")
        torch.save(self._model.state_dict(), path)
        logging.info("Done")

    def load_model(self, path: Optional[str] = "saves/model.pt") -> None:
        logging.info("Loading model weights...")
        self._model.load_state_dict(torch.load(path))
        logging.info("Done")