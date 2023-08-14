from torch.utils.tensorboard import SummaryWriter
import numpy as np

from my_chess.nn.pytorch_nn.AlphaChess.ValAccuracyBase import AlphaValHead
from shared.shared_functionality import value_to_outcome


class AlphaValValue(AlphaValHead):
    def __init__(self, writer: SummaryWriter, tensorboard: str, epoch: int):
        super().__init__(writer, tensorboard, epoch)
        self.tp = 0.0
        self.fp = 0.0

    def update_accuracy_from_batch(self, labels: np.ndarray, outputs: np.ndarray) -> None:
        l2 = value_to_outcome(labels)
        o2 = value_to_outcome(outputs[:, 0])
        tp = ((l2 == o2).sum())
        fp = ((l2 != o2).sum())
        self.tp += tp
        self.fp += fp

    def print_and_log_accuracy(self) -> float:
        epoch_acc = float(self.tp) / (self.tp + self.fp)
        print(f'Val Precision: {round(epoch_acc, 3)} ')
        if self.tensorboard == 'on':
            self.writer.add_scalar(f'Val Precision Value', epoch_acc, self.epoch)
        return epoch_acc
