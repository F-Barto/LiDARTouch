#! /bin/bash

NAME=pac_guiding_scratch_photo_ranger_cosannealstart=25e-2_lr=9e-5_bs=8_resnet18_`date +"%m%d%y_%H%M%S"`; \
python ./submit_job.py -n $NAME --gpumem 16000 \
-pf ../configs/projects/monocular_frontcam_depth_argo.yml \
-po "experiment_name=$NAME" \
-mf ../configs/experiments/argoverse/self_supervised_frontcam.yml \
-mo "model=pac_guiding_resnet18 dataloaders.train.batch_size=1 dataloaders.val.batch_size=5 dataloaders.test.batch_size=5 trainer.accumulate_grad_batches=8"

NAME=pac_guiding_scratch_photo+vel_ranger_cosannealstart=25e-2_lr=9e-5_bs=8_resnet18_`date +"%m%d%y_%H%M%S"`; \
python ./submit_job.py -n $NAME --gpumem 16000 \
-pf ../configs/projects/monocular_frontcam_depth_argo.yml \
-po "experiment_name=$NAME" \
-mf ../configs/experiments/argoverse/self_supervised_frontcam.yml \
-mo "model=pac_guiding_resnet18 losses=velocity_loss datasets.train.load_pose=True dataloaders.train.batch_size=1 dataloaders.val.batch_size=5 dataloaders.test.batch_size=5 trainer.accumulate_grad_batches=8 "

### DDAD pre-trained

NAME=pac_guiding_pretrained_D_photo_ranger_cosannealstart=25e-2_lr=9e-5_bs=8_resnet18_`date +"%m%d%y_%H%M%S"`; \
python ./submit_job.py -n $NAME --gpumem 16000 \
-pf ../configs/projects/monocular_frontcam_depth_argo.yml \
-po "experiment_name=$NAME" \
-mf ../configs/experiments/argoverse/self_supervised_frontcam.yml \
-mo "model=pac_guiding_resnet18 model.tri_checkpoint_path='/home/clear/fbartocc/depth_project/Depth/weights/ResNet18_MR_selfsup_D.ckpt' dataloaders.train.batch_size=1 dataloaders.val.batch_size=5 dataloaders.test.batch_size=5 trainer.accumulate_grad_batches=8"

NAME=pac_guiding_pretrained_D_photo+vel_ranger_cosannealstart=25e-2_lr=9e-5_bs=8_resnet18_`date +"%m%d%y_%H%M%S"`; \
python ./submit_job.py -n $NAME --gpumem 16000 \
-pf ../configs/projects/monocular_frontcam_depth_argo.yml \
-po "experiment_name=$NAME" \
-mf ../configs/experiments/argoverse/self_supervised_frontcam.yml \
-mo "model=pac_guiding_resnet18 model.tri_checkpoint_path='/home/clear/fbartocc/depth_project/Depth/weights/ResNet18_MR_selfsup_D.ckpt' losses=velocity_loss datasets.train.load_pose=True dataloaders.train.batch_size=1 dataloaders.val.batch_size=5 dataloaders.test.batch_size=5 trainer.accumulate_grad_batches=8"

NAME=pac_guiding_pretrained_D_reproj_ranger_cosannealstart=25e-2_lr=9e-5_bs=8_resnet18_`date +"%m%d%y_%H%M%S"`; \
python ./submit_job.py -n $NAME --gpumem 16000 \
-pf ../configs/projects/monocular_frontcam_depth_argo.yml \
-po "experiment_name=$NAME" \
-mf ../configs/experiments/argoverse/self_supervised_frontcam.yml \
-mo "model=pac_guiding_resnet18 model.tri_checkpoint_path='/home/clear/fbartocc/depth_project/Depth/weights/ResNet18_MR_selfsup_D.ckpt' losses=semi_supervised dataloaders.train.batch_size=1 dataloaders.val.batch_size=5 dataloaders.test.batch_size=5 trainer.accumulate_grad_batches=8"

NAME=pac_guiding_pretrained_D_hinted_ranger_cosannealstart=25e-2_lr=9e-5_bs=8_resnet18_`date +"%m%d%y_%H%M%S"`; \
python ./submit_job.py -n $NAME --gpumem 16000 \
-pf ../configs/projects/monocular_frontcam_depth_argo.yml \
-po "experiment_name=$NAME" \
-mf ../configs/experiments/argoverse/self_supervised_frontcam.yml \
-mo "model=pac_guiding_resnet18 model.tri_checkpoint_path='/home/clear/fbartocc/depth_project/Depth/weights/ResNet18_MR_selfsup_D.ckpt' losses=hinted_loss dataloaders.train.batch_size=1 dataloaders.val.batch_size=5 dataloaders.test.batch_size=5 trainer.accumulate_grad_batches=8"


