from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class AlphaValHead(ABC):
    @abstractmethod
    def __init__(self, writer: SummaryWriter, tensorboard: str, epoch: int):
        self.writer = writer
        self.tensorboard = tensorboard
        self.epoch = epoch
    @abstractmethod
    def update_accuracy_from_batch(self, labels: np.ndarray, outputs: np.ndarray) -> None:
        pass
    @abstractmethod
    def print_and_log_accuracy(self) -> float:
        pass

