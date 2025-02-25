import logging
import os

import torch
import torch.nn as nn


class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()

    def save_model_weights(self, f_path: str = "saves/model.pt") -> None:
        logging.info("Saving model weights...")
        os.makedirs(os.path.dirname(f_path), exist_ok=True)
        torch.save(self.state_dict(), f_path)
        logging.info("Model weights saved successfully.")

    def load_model_weights(
        self, f_path: str = "saves/model.pt", map_location: str = "cpu"
    ) -> None:
        logging.info("Loading model weights...")
        self.load_state_dict(torch.load(f_path, map_location=map_location))
        logging.info("Model weights loaded successfully.")
