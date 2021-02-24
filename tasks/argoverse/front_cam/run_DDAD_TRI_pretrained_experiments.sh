#! /bin/bash

NAME=pretrained_D_photo_ranger_cosannealstart=25e-2_lr=9e-5_bs=8_resnet18_`date +"%m%d%y_%H%M%S"`; \
python ./submit_job.py -n $NAME --gpumem 13000 \
-pf ../configs/projects/monocular_frontcam_depth_argo.yml \
-po "experiment_name=$NAME" \
-mf ../configs/experiments/argoverse/self_supervised_frontcam.yml \
-mo "model.tri_checkpoint_path='/home/clear/fbartocc/depth_project/Depth/weights/ResNet18_MR_selfsup_D.ckpt'"

NAME=pretrained_D_photo+vel_ranger_cosannealstart=25e-2_lr=9e-5_bs=8_resnet18_`date +"%m%d%y_%H%M%S"`; \
python ./submit_job.py -n $NAME --gpumem 13000 \
-pf ../configs/projects/monocular_frontcam_depth_argo.yml \
-po "experiment_name=$NAME" \
-mf ../configs/experiments/argoverse/self_supervised_frontcam.yml \
-mo "model.tri_checkpoint_path='/home/clear/fbartocc/depth_project/Depth/weights/ResNet18_MR_selfsup_D.ckpt' losses=velocity_loss datasets.train.load_pose=True"

NAME=pretrained_D_reproj_ranger_cosannealstart=25e-2_lr=9e-5_bs=8_resnet18_`date +"%m%d%y_%H%M%S"`; \
python ./submit_job.py -n $NAME --gpumem 13000 \
-pf ../configs/projects/monocular_frontcam_depth_argo.yml \
-po "experiment_name=$NAME" \
-mf ../configs/experiments/argoverse/self_supervised_frontcam.yml \
-mo "model.tri_checkpoint_path='/home/clear/fbartocc/depth_project/Depth/weights/ResNet18_MR_selfsup_D.ckpt' losses=semi_supervised"

NAME=pretrained_D_hinted_ranger_cosannealstart=25e-2_lr=9e-5_bs=8_resnet18_`date +"%m%d%y_%H%M%S"`; \
python ./submit_job.py -n $NAME --gpumem 13000 \
-pf ../configs/projects/monocular_frontcam_depth_argo.yml \
-po "experiment_name=$NAME" \
-mf ../configs/experiments/argoverse/self_supervised_frontcam.yml \
-mo "model.tri_checkpoint_path='/home/clear/fbartocc/depth_project/Depth/weights/ResNet18_MR_selfsup_D.ckpt' losses=hinted_loss"

################## RESNET 50

NAME=pretrained_D_photo+vel_ranger_cosannealstart=25e-2_lr=9e-5_bs=8_resnet50_`date +"%m%d%y_%H%M%S"`; \
python ./submit_job.py -n $NAME --gpumem 16000 \
-pf ../configs/projects/monocular_frontcam_depth_argo.yml \
-po "experiment_name=$NAME" \
-mf ../configs/experiments/argoverse/self_supervised_frontcam.yml \
-mo "model.tri_checkpoint_path='/home/clear/fbartocc/depth_project/Depth/weights/ResNet18_MR_selfsup_D.ckpt' losses=velocity_loss model=monodepth50 datasets.train.load_pose=True"

################## mish activation

NAME=pretrained_D_mish_photo_ranger_cosannealstart=25e-2_lr=9e-5_bs=8_resnet18_`date +"%m%d%y_%H%M%S"`; \
python ./submit_job.py -n $NAME --gpumem 13000 \
-pf ../configs/projects/monocular_frontcam_depth_argo.yml \
-po "experiment_name=$NAME" \
-mf ../configs/experiments/argoverse/self_supervised_frontcam.yml \
-mo "model.tri_checkpoint_path='/home/clear/fbartocc/depth_project/Depth/weights/ResNet18_MR_selfsup_D.ckpt' model.depth_net.options.activation=mish"

