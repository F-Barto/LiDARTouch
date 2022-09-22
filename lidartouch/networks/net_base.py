import torch.nn as nn

class NetworkBase(nn.Module):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert isinstance(self.required_inputs, list)


    @property
    def required_inputs(self):
        raise NotImplementedError


