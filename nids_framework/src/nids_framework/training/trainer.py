import logging
from typing import Optional, Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .metrics.base import Metric
from .early_stopping import EarlyStopping


class Trainer:

    __slots__ = ["model",
                 "criterion",
                 "optimizer",
                 "device",
                 "early_stopping",]

    def __init__(self, criterion: Optional[nn.Module] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 patience: Optional[int] = None, delta: Optional[float] = None, device: Optional[str] = "cpu") -> None:
        self.model = None
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        self.early_stopping : Optional[EarlyStopping] = None
        if patience is not None and delta is not None:
            self.early_stopping = EarlyStopping(patience=patience, delta=delta)

    def set_model(self, model: nn.Module) -> None:
        self.model = model

    def _step_one_batch(self, batch: Tuple, metric: Optional[Metric] = None) -> torch.Tensor:
        if len(batch) == 2:
            inputs, labels = batch
            inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
        else:
            inputs, num_target, cat_target = batch
            inputs, num_target, cat_target = inputs.to(self.device, non_blocking=True), num_target.to(self.device, non_blocking=True), cat_target.to(self.device, non_blocking=True)

        outputs = self.model(inputs)

        if metric and len(batch) == 2: metric.step(outputs, labels)

        if len(batch) == 2:
            return self.criterion(outputs, labels)
        else:
            num_recon, cat_recon = outputs
            return self.criterion(num_recon, num_target, cat_recon, cat_target)

    def train_one_epoch(self, data_loader: DataLoader, epoch_steps: Optional[int] = None) -> float:
        self.model.train()
        epoch_loss = torch.tensor(0.0, device=self.device)
        total_steps = min(epoch_steps, len(data_loader)) if epoch_steps else len(data_loader)
        data_iter = iter(data_loader)

        for _ in tqdm(range(total_steps), desc="Training"):
            batch = next(data_iter)
            self.optimizer.zero_grad()
            loss = self._step_one_batch(batch)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.detach()

        return epoch_loss.item() / total_steps

    def train(self, n_epoch: int, data_loader: DataLoader, epoch_steps: Optional[int] = None,
              epochs_until_validation: Optional[int] = 1, valid_data_loader: Optional[DataLoader] = None) -> float:
        if self.model is None:
            raise ValueError("Model is not set. Call `set_model()` to assign a model.")

        logging.info(f"Starting {n_epoch}-epoch training loop...")
        total_train_loss = 0.0

        for epoch in range(n_epoch):
            epoch_loss = self.train_one_epoch(data_loader, epoch_steps)
            total_train_loss += epoch_loss
            logging.info(f"Epoch {epoch + 1}/{n_epoch} Loss: {epoch_loss:.6f}")

            if valid_data_loader and self.early_stopping and (epoch + 1) % epochs_until_validation == 0:
                validation_loss = self.validate(valid_data_loader)
                self.early_stopping(validation_loss, self.model)

                if self.early_stopping.early_stop:
                    logging.info(f"Early stopping at epoch {epoch + 1}")
                    logging.info(f"Training completed early. Average Loss: {total_train_loss / (epoch + 1):.6f}")
                    self.early_stopping.restore_best_weights(self.model)
                    break

        average_train_loss = total_train_loss / n_epoch
        logging.info(f"Training completed: {n_epoch} epochs, Average Loss: {average_train_loss:.6f}")
        return average_train_loss

    def validate(self, data_loader: DataLoader) -> float:
        logging.info("Starting validation loop...")
        self.model.eval()
        validation_loss = torch.tensor(0.0, device=self.device)

        with torch.inference_mode():
            for batch in tqdm(data_loader, desc="Validating"):
                loss = self._step_one_batch(batch)
                validation_loss += loss.detach()

        validation_loss /= len(data_loader)
        logging.info(f"Validation loss: {validation_loss.item():.6f}")
        return validation_loss

    def test(self, data_loader: DataLoader, metric: Optional[Metric] = None) -> None:
        logging.info("Starting test loop...")
        self.model.eval()
        test_loss = torch.tensor(0.0, device=self.device)

        with torch.inference_mode():
            for batch in tqdm(data_loader, desc="Testing"):
                loss = self._step_one_batch(batch, metric)
                test_loss += loss.detach()

        test_loss /= len(data_loader)
        logging.info(f"Test loss: {test_loss.item():.6f}")

        if metric:
            metric.compute_metrics()
            logging.info(f"{metric}\n")
