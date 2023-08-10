from torch.utils.tensorboard import SummaryWriter
import numpy as np

from my_chess.nn.pytorch_nn.AlphaChess.ValAccuracyBase import AlphaValHead
from shared.shared_functionality import value_to_outcome, get_config


class AlphaValPolicy(AlphaValHead):
    def __init__(self, writer: SummaryWriter, tensorboard: str, epoch: int):
        super().__init__(writer, tensorboard, epoch)
        self.top_k = 3
        config = get_config()
        self.policy_loss_move_weight = config['train']['torch']['policy_loss_move_weight']
        self.tp = np.zeros(self.top_k, dtype=float)
        self.fp = np.zeros(self.top_k, dtype=float)

    def update_accuracy_from_batch(self, labels: np.ndarray, outputs: np.ndarray) -> None:
        policy_labels = labels[0]
        masks_labels = labels[1]
        masks_labels[masks_labels<0.5] = self.policy_loss_move_weight
        o_order = np.argsort(outputs, axis=1)
        for j in range(policy_labels.shape[0]):
            l_non_zero_indices = np.where(policy_labels[j] > 0)[0]
            for k in range(1, self.top_k + 1):
                if set(o_order[j, -k:]).intersection(l_non_zero_indices):
                    self.tp[k - 1] += 1 * masks_labels[j]
                else:
                    self.fp[k - 1] += 1 * masks_labels[j]

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
