#! /bin/bash

NAME=scratch_photo_ranger_cosannealstart=25e-2_lr=9e-5_bs=8_resnet18_`date +"%m%d%y_%H%M%S"`; \
python ./submit_job.py -n $NAME --gpumem 13000 \
-pf ../configs/projects/monocular_randcam_depth_argo.yml \
-po "experiment_name=$NAME" \
-mf ../configs/experiments/argoverse/self_supervised_randcam.yml

NAME=scratch_photo+vel_ranger_cosannealstart=25e-2_lr=9e-5_bs=8_resnet18_`date +"%m%d%y_%H%M%S"`; \
python ./submit_job.py -n $NAME --gpumem 13000 \
-pf ../configs/projects/monocular_randcam_depth_argo.yml \
-po "experiment_name=$NAME" \
-mf ../configs/experiments/argoverse/self_supervised_randcam.yml \
-mo "losses=velocity_loss datasets.train.load_pose=True"

NAME=scratch_reproj_ranger_cosannealstart=25e-2_lr=9e-5_bs=8_resnet18_`date +"%m%d%y_%H%M%S"`; \
python ./submit_job.py -n $NAME --gpumem 13000 \
-pf ../configs/projects/monocular_randcam_depth_argo.yml \
-po "experiment_name=$NAME" \
-mf ../configs/experiments/argoverse/self_supervised_randcam.yml \
-mo "losses=semi_supervised"

NAME=scratch_hinted_ranger_cosannealstart=25e-2_lr=9e-5_bs=8_resnet18_`date +"%m%d%y_%H%M%S"`; \
python ./submit_job.py -n $NAME --gpumem 13000 \
-pf ../configs/projects/monocular_randcam_depth_argo.yml \
-po "experiment_name=$NAME" \
-mf ../configs/experiments/argoverse/self_supervised_randcam.yml \
-mo "losses=hinted_loss"

################## RESNET 50

NAME=scratch_photo+vel_ranger_cosannealstart=25e-2_lr=9e-5_bs=8_resnet50_`date +"%m%d%y_%H%M%S"`; \
python ./submit_job.py -n $NAME --gpumem 16000 \
-pf ../configs/projects/monocular_randcam_depth_argo.yml \
-po "experiment_name=$NAME" \
-mf ../configs/experiments/argoverse/self_supervised_randcam.yml \
-mo "losses=velocity_loss model=monodepth50  datasets.train.load_pose=True"

################## Bigger batch

