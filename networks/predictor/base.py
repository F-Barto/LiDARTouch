import abc
import torch.nn as nn

class MultiScaleBasePredictor(nn.Module):
    def __init__(self, scales):
        super(MultiScaleBasePredictor, self).__init__()
        self.scales = scales

    @abc.abstractmethod
    def forward(self, x, i):
        pass

    @abc.abstractmethod
    def get_prediction(self, i):
        pass

    @abc.abstractmethod
    def compile_predictions(self):
        pass