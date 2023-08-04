from torch.utils.tensorboard import SummaryWriter
import numpy as np

from my_chess.nn.pytorch_nn.AlphaChess.ValAccuracyBase import AlphaValHead
from shared.shared_functionality import value_to_outcome


class AlphaValPolicy(AlphaValHead):
    def __init__(self, writer: SummaryWriter, tensorboard: str, epoch: int):
        super().__init__(writer, tensorboard, epoch)
        self.top_k = 3
        self.tp = np.zeros(self.top_k, dtype=int)
        self.fp = np.zeros(self.top_k, dtype=int)

    def update_accuracy_from_batch(self, labels: np.ndarray, outputs: np.ndarray) -> None:
        o_order = np.argsort(outputs, axis=1)
        for j in range(labels.shape[0]):
            l_non_zero_indices = np.where(labels[j] > 0)[0]
            for k in range(1, self.top_k + 1):
                if set(o_order[j, -k:]).intersection(l_non_zero_indices):
                    self.tp[k - 1] += 1
                else:
                    self.fp[k - 1] += 1

    def print_and_log_accuracy(self) -> float:
        epoch_acc = 0.0
        for k in range(1, self.top_k + 1):
            precision = float(self.tp[k - 1]) / (self.tp[k - 1] + self.fp[k - 1])
            if k == 1:
                epoch_acc = precision
            print(f'Policy k : {k} precision: {round(precision, 3)} ')
            if self.tensorboard == 'on':
                self.writer.add_scalar(f"Val precision Policy k={k}", precision, self.epoch)
        return epoch_acc
