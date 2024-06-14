from typing import Tuple, Optional
import logging
from tqdm import tqdm

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
        "_device",
    ]

    def __init__(self, model: nn.Module, criterion: _Loss, optimizer: Optimizer) -> None:
        self._model: nn.Module = model
        self._criterion: _Loss = criterion
        self._optimizer: Optimizer = optimizer
        self._device: str = ('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, n_epoch: int, data_loader: DataLoader, epoch_steps: Optional[int] = None) -> None:
        logging.info(f"Starting {n_epoch}-epoch training loop...")
        train_loss = 0.0
        self._model.train()

        for epoch in range(n_epoch):
            epoch_loss = 0.0
            
            if epoch_steps is None:
                for batch in tqdm(data_loader):
                    epoch_loss += self.one_batch(batch)
                epoch_loss /= len(data_loader)
            else:
                if epoch_steps > len(data_loader): raise ValueError(f"Epoch steps must be less or at least equal to {len(self._data_loader)}.")
                epoch_iter = iter(data_loader)
                for _ in tqdm(range(epoch_steps)):
                    batch = next(epoch_iter)
                    epoch_loss += self.fit_one_batch(batch)
                epoch_loss /= epoch_steps
            
            train_loss += epoch_loss
            logging.info(f"Epoch: {epoch} Loss: {epoch_loss:.6f}.\n")
        
        train_loss /= n_epoch
        logging.info("Done with training.")
        logging.info(f"Trained for {n_epoch} epochs with loss: {train_loss:.6f}.\n")


    def fit_one_batch(self, batch: Tuple) -> None:
        inputs, labels = batch
        inputs = inputs.to(self._device) 
        labels = labels.to(self._device)

        self._optimizer.zero_grad()
        outputs = self._model(inputs)

        loss = self._criterion(outputs, labels) 
        loss.backward()
        self._optimizer.step()

        return loss.item()

    def test(self, data_loader: DataLoader):
        logging.info(f"Starting test loop...")
        num_correct = 0.0
        self._model.eval()

        for batch in tqdm(data_loader):
            num_correct += self.test_one_batch(batch)

        accuracy = num_correct / (len(data_loader) * data_loader.batch_size)

        logging.info("Done with testing.")
        logging.info(f"Test accuracy: {accuracy:.6f}.\n")

    def test_one_batch(self, batch: Tuple):
        inputs, labels = batch
        inputs = inputs.to(self._device) 
        labels = labels.to(self._device)

        outputs = self._model(inputs)
        predicted = torch.where(outputs >= 0.5, torch.tensor(1.0), torch.tensor(0.0))
        num_correct = (predicted == labels).float().sum()
        
        return num_correct

    def save_model(self) -> None:
        pass

    def load_model(self) -> None:
        pass