from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
import math

class FlatCosAnnealScheduler(_LRScheduler):

    def __init__(self, optimizer, step_factor, len_dataset, max_epochs, anneal_start=0.7, anneal_strategy='cos', last_epoch=-1,
                 min_lr=0.):

        self.validate_params(optimizer, anneal_strategy, anneal_start)

        self.optimizer = optimizer

        if len_dataset % step_factor != 0:
            len_dataset += 1

        self.total_steps = (len_dataset  // step_factor) * max_epochs

        self.nb_steps_flat = int(self.total_steps * anneal_start) # the number of steps the lr curve is flat
        self.nb_steps_anneal = self.total_steps - self.nb_steps_flat # the number of steps the lr curve is annealing

        # Initialize learning rate variables
        if last_epoch == -1:
            for group in self.optimizer.param_groups:
                group['min_lr'] = min_lr

        self.last_epoch = last_epoch
        super(FlatCosAnnealScheduler, self).__init__(optimizer, last_epoch)

    def validate_params(self, optimizer, anneal_strategy, anneal_start):

        # Validate optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))

        # Validate anneal_strategy
        if anneal_strategy == 'cos':
            self.anneal_func = self._annealing_cos
        elif anneal_strategy == 'linear':
            self.anneal_func = self._annealing_linear
        else:
            raise ValueError("anneal_strategy must by one of 'cos' or 'linear', instead got {}".format(anneal_strategy))

        # Validate pct_start
        if anneal_start < 0 or anneal_start > 1 or not isinstance(anneal_start, float):
            raise ValueError(f"Expected float between 0 and 1 for anneal_start, but got {anneal_start}")

    def _annealing_cos(self, start, end, pct):
        "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        cos_out = math.cos(pct * math.pi) + 1
        return end + (start - end) / 2.0 * cos_out

    def _annealing_linear(self, start, end, pct):
        "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        return (end - start) * pct + start

    def get_lr(self):

        # self.last_epoch is in fact last_step as schedulers.step() should be called every step for this schedulers
        step_num = self.last_epoch

        if step_num > self.total_steps:
            raise ValueError("Tried to step {} times. The specified number of total steps is {}"
                             .format(step_num + 1, self.total_steps))

        if step_num <= self.nb_steps_flat:
            return list(self.base_lrs)
        else:
            lrs = []
            pct_progression_annealing = (self.nb_steps_flat - step_num) / self.nb_steps_anneal
            for group in self.optimizer.param_groups:
                computed_lr = self.anneal_func(group['initial_lr'], group['min_lr'], pct_progression_annealing)
                lrs.append(computed_lr)

            return lrs
