from pytorch_lightning import _logger as terminal_logger

from losses.elements.hinted_loss import HintedLoss
from losses.elements.photometric_loss import PhotometricLoss
from losses.elements.smoothness_loss import SmoothnessLoss
from losses.elements.supervised_loss import SupervisedLoss
from losses.elements.velocity_loss import VelocityLoss

class LossHandler(object):

    def __init__(self, losses_hparams, **kwargs):
        super().__init__(**kwargs)

        self.losses_hparams = losses_hparams

        self._losses_associations = {
            'hinted': HintedLoss,
            'regression': SupervisedLoss,
            'photo': PhotometricLoss,
            'velocity': VelocityLoss,
            'smoothness': SmoothnessLoss
        }

    def parse_all_losses(self):
        losses_names = list(self._losses_associations.keys())
        return self.parse_losses(losses_names)

    def parse_losses(self, losses_names):

        losses = {}

        for loss_name in losses_names:
            if loss_name in self.losses_hparams:
                loss_parameters = self.losses_hparams.pop(loss_name)
                loss_object = self._losses_associations[loss_name](**loss_parameters)

                losses[loss_name] = loss_object

                terminal_logger.info(f"Initialized loss '{loss_name}' with parameters {loss_parameters}")

        return losses