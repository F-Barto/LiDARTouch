from lidartouch.models.base_sfm_depth import BaseSfMDepth
from lidartouch.utils.identifiers import TARGET_VIEW, SOURCE_VIEWS, SPARSE_DEPTH
from lidartouch.utils.pose import Pose

import hydra

class SfMDepthPoseNet(BaseSfMDepth):

    def __init__(self, pose_net: dict, posenet_optimizer_conf: dict, posenet_scheduler_conf: dict,
                 posenet_scheduler_interval: str = 'epoch', *args, **kwargs):
        """

        """
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.posenet_optimizer_conf = posenet_optimizer_conf
        self.posenet_scheduler_conf = posenet_scheduler_conf
        self.posenet_scheduler_interval = posenet_scheduler_interval

        self.pose_net = hydra.utils.instantiate(pose_net)


    def configure_optimizers(self):
        depth_net_optimization = super().configure_optimizers()

        if not self.posenet_optimizer_conf:
            return None

        optimizer = hydra.utils.instantiate(self.posenet_optimizer_conf, params=self.pose_net.parameters())

        if not self.posenet_scheduler_conf:
            return (depth_net_optimization, optimizer)

        scheduler = hydra.utils.instantiate(self.posenet_scheduler_conf, optimizer=optimizer)

        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": self.posenet_scheduler_interval,
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            # "monitor": "val_loss",
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            # "strict": True,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": 'lr_posenet',
        }

        pose_net_optimization = {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

        return (depth_net_optimization, pose_net_optimization)

    def compute_poses(self, batch):
        """Compute poses from image and a sequence of context images"""
        pose_vec = self.pose_net(batch[TARGET_VIEW], batch[SOURCE_VIEWS], sparse_depth=batch[SPARSE_DEPTH])
        poses = [Pose.from_vec(pose_vec[:, i], 'euler') for i in range(pose_vec.shape[1])]
        return {'poses': poses}



