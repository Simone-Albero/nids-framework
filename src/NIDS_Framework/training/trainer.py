from typing import Tuple, Optional
import logging

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

class Trainer:

    __slots__ = [
        "_model",
        "_criterion",
        "_optimizer",
        "_data_loader",
        "_device",
    ]

    def __init__(self, model: nn.Module, criterion: _Loss, optimizer: Optimizer, data_loader: DataLoader) -> None:
        self._model: nn.Module = model
        self._criterion: _Loss = criterion
        self._optimizer: Optimizer = optimizer
        self._data_loader: DataLoader = data_loader
        self._device: str = ('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, n_epoch: int, epoch_steps: Optional[int] = None) -> None:
        logging.info(f"Starting {n_epoch} epoch trainig loop...")
        for epoch in range(n_epoch):
            epoch_loss = 0.0
            
            if epoch_steps is None:
                for batch in self._data_loader:
                    epoch_loss += self.one_batch(batch)
            else:
                if epoch_steps > len(self._data_loader): raise ValueError(f"Epoch steps must be less or at least equal to {len(self._data_loader)}.")
                epoch_iter = iter(self._data_loader)
                for _ in range(epoch_steps):
                    batch = next(epoch_iter)
                    epoch_loss += self.one_batch(batch)

            logging.info(f"Epoch: {epoch} Loss: {epoch_loss/len(self._data_loader)}")


    def one_batch(self, batch: Tuple) -> None:
        inputs, labels = batch
        inputs = inputs.to(self._device) 
        labels = labels.to(self._device)

        self._optimizer.zero_grad()
        outputs = self._model(inputs)

        loss = self._criterion(outputs, labels) 
        loss.backward()
        self._optimizer.step()

        return loss.item()

    def save_model(self) -> None:
        pass

    def load_model(self) -> None:
        pass