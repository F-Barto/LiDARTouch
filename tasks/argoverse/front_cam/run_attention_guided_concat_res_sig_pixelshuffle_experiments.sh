#! /bin/bash




NAME=AGcat-mult-sig_scratch_photo_ranger_cosannealstart=25e-2_lr=9e-5_bs=8_PS_resnet18_`date +"%m%d_%H%M%S"`; \
python ./submit_job.py -n $NAME --gpumem 16000 \
-pf ../configs/projects/monocular_frontcam_depth_argo.yml \
-po "experiment_name=$NAME" \
-mf ../configs/experiments/argoverse/self_supervised_frontcam.yml \
-mo "model=attention_guiding_concat_mult_sig_upshuffle_resnet18 dataloaders.train.batch_size=1 dataloaders.val.batch_size=5 dataloaders.test.batch_size=5 trainer.accumulate_grad_batches=8"

NAME=AGcat-mult-sig_scratch_photo+vel_ranger_cosannealstart=25e-2_lr=9e-5_bs=8_PS_resnet18_`date +"%m%d_%H%M%S"`; \
python ./submit_job.py -n $NAME --gpumem 16000 \
-pf ../configs/projects/monocular_frontcam_depth_argo.yml \
-po "experiment_name=$NAME" \
-mf ../configs/experiments/argoverse/self_supervised_frontcam.yml \
-mo "model=attention_guiding_concat_mult_sig_upshuffle_resnet18 losses=velocity_loss datasets.train.load_pose=True dataloaders.train.batch_size=1 dataloaders.val.batch_size=5 dataloaders.test.batch_size=5 trainer.accumulate_grad_batches=8 "

NAME=AGcat-mult-sig_scratch_hinted_reproj_ranger_cosannealstart=25e-2_lr=9e-5_bs=8_PS_resnet18_`date +"%m%d_%H%M%S"`; \
python ./submit_job.py -n $NAME --gpumem 16000 \
-pf ../configs/projects/monocular_frontcam_depth_argo.yml \
-po "experiment_name=$NAME" \
-mf ../configs/experiments/argoverse/self_supervised_frontcam.yml \
-mo "model=attention_guiding_concat_mult_sig_upshuffle_resnet18 losses=hinted_loss dataloaders.train.batch_size=1 dataloaders.val.batch_size=5 dataloaders.test.batch_size=5 trainer.accumulate_grad_batches=8"

NAME=AGcat-mult-sig_scratch_hinted_berhu_ranger_cosannealstart=25e-2_lr=9e-5_bs=8_PS_resnet18_`date +"%m%d_%H%M%S"`; \
python ./submit_job.py -n $NAME --gpumem 16000 \
-pf ../configs/projects/monocular_frontcam_depth_argo.yml \
-po "experiment_name=$NAME" \
-mf ../configs/experiments/argoverse/self_supervised_frontcam.yml \
-mo "model=attention_guiding_concat_mult_sig_upshuffle_resnet18 losses=hinted_loss losses.HintedMultiViewPhotometricLoss.supervised_method='sparse-berhu' dataloaders.train.batch_size=1 dataloaders.val.batch_size=5 dataloaders.test.batch_size=5 trainer.accumulate_grad_batches=8"

NAME=AGcat-mult-sig_scratch_hinted_l1_ranger_cosannealstart=25e-2_lr=9e-5_bs=8_PS_resnet18_`date +"%m%d_%H%M%S"`; \
python ./submit_job.py -n $NAME --gpumem 16000 \
-pf ../configs/projects/monocular_frontcam_depth_argo.yml \
-po "experiment_name=$NAME" \
-mf ../configs/experiments/argoverse/self_supervised_frontcam.yml \
-mo "model=attention_guiding_concat_mult_sig_upshuffle_resnet18 losses=hinted_loss losses.HintedMultiViewPhotometricLoss.supervised_method='sparse-l1' dataloaders.train.batch_size=1 dataloaders.val.batch_size=5 dataloaders.test.batch_size=5 trainer.accumulate_grad_batches=8"


### DDAD pre-trained

NAME=AGcat-mult-sig_pretrained_D_photo_ranger_cosannealstart=25e-2_lr=9e-5_bs=8_PS_resnet18_`date +"%m%d_%H%M%S"`; \
python ./submit_job.py -n $NAME --gpumem 16000 \
-pf ../configs/projects/monocular_frontcam_depth_argo.yml \
-po "experiment_name=$NAME" \
-mf ../configs/experiments/argoverse/self_supervised_frontcam.yml \
-mo "model=attention_guiding_concat_mult_sig_upshuffle_resnet18 model.tri_checkpoint_path='/home/clear/fbartocc/depth_project/Depth/weights/ResNet18_MR_selfsup_D.ckpt' dataloaders.train.batch_size=1 dataloaders.val.batch_size=5 dataloaders.test.batch_size=5 trainer.accumulate_grad_batches=8"

NAME=AGcat-mult-sig_pretrained_D_photo+vel_ranger_cosannealstart=25e-2_lr=9e-5_bs=8_PS_resnet18_`date +"%m%d_%H%M%S"`; \
python ./submit_job.py -n $NAME --gpumem 16000 \
-pf ../configs/projects/monocular_frontcam_depth_argo.yml \
-po "experiment_name=$NAME" \
-mf ../configs/experiments/argoverse/self_supervised_frontcam.yml \
-mo "model=attention_guiding_concat_mult_sig_upshuffle_resnet18 model.tri_checkpoint_path='/home/clear/fbartocc/depth_project/Depth/weights/ResNet18_MR_selfsup_D.ckpt' losses=velocity_loss datasets.train.load_pose=True dataloaders.train.batch_size=1 dataloaders.val.batch_size=5 dataloaders.test.batch_size=5 trainer.accumulate_grad_batches=8"

NAME=AGcat-mult-sig_pretrained_D_reproj_ranger_cosannealstart=25e-2_lr=9e-5_bs=8_PS_resnet18_`date +"%m%d_%H%M%S"`; \
python ./submit_job.py -n $NAME --gpumem 16000 \
-pf ../configs/projects/monocular_frontcam_depth_argo.yml \
-po "experiment_name=$NAME" \
-mf ../configs/experiments/argoverse/self_supervised_frontcam.yml \
-mo "model=attention_guiding_concat_mult_sig_upshuffle_resnet18 model.tri_checkpoint_path='/home/clear/fbartocc/depth_project/Depth/weights/ResNet18_MR_selfsup_D.ckpt' dataloaders.train.batch_size=1 dataloaders.val.batch_size=5 dataloaders.test.batch_size=5 trainer.accumulate_grad_batches=8 losses=semi_supervised"

NAME=AGcat-mult-sig_pretrained_D_hinted_ranger_cosannealstart=25e-2_lr=9e-5_bs=8_PS_resnet18_`date +"%m%d_%H%M%S"`; \
python ./submit_job.py -n $NAME --gpumem 16000 \
-pf ../configs/projects/monocular_frontcam_depth_argo.yml \
-po "experiment_name=$NAME" \
-mf ../configs/experiments/argoverse/self_supervised_frontcam.yml \
-mo "model=attention_guiding_concat_mult_sig_upshuffle_resnet18 model.tri_checkpoint_path='/home/clear/fbartocc/depth_project/Depth/weights/ResNet18_MR_selfsup_D.ckpt' losses=hinted_loss dataloaders.train.batch_size=1 dataloaders.val.batch_size=5 dataloaders.test.batch_size=5 trainer.accumulate_grad_batches=8"

NAME=AGcat-mult-sig_pretrained_D_hinted_reproj_ranger_cosannealstart=25e-2_lr=9e-5_bs=8_PS_resnet18_`date +"%m%d_%H%M%S"`; \
python ./submit_job.py -n $NAME --gpumem 16000 \
-pf ../configs/projects/monocular_frontcam_depth_argo.yml \
-po "experiment_name=$NAME" \
-mf ../configs/experiments/argoverse/self_supervised_frontcam.yml \
-mo "model=attention_guiding_concat_mult_sig_upshuffle_resnet18 model.tri_checkpoint_path='/home/clear/fbartocc/depth_project/Depth/weights/ResNet18_MR_selfsup_D.ckpt' losses=hinted_loss dataloaders.train.batch_size=1 dataloaders.val.batch_size=5 dataloaders.test.batch_size=5 trainer.accumulate_grad_batches=8"

NAME=AGcat-mult-sig_pretrained_D_hinted_berhu_ranger_cosannealstart=25e-2_lr=9e-5_bs=8_PS_resnet18_`date +"%m%d_%H%M%S"`; \
python ./submit_job.py -n $NAME --gpumem 16000 \
-pf ../configs/projects/monocular_frontcam_depth_argo.yml \
-po "experiment_name=$NAME" \
-mf ../configs/experiments/argoverse/self_supervised_frontcam.yml \
-mo "model=attention_guiding_concat_mult_sig_upshuffle_resnet18 model.tri_checkpoint_path='/home/clear/fbartocc/depth_project/Depth/weights/ResNet18_MR_selfsup_D.ckpt' losses=hinted_loss losses.HintedMultiViewPhotometricLoss.supervised_method='sparse-berhu' dataloaders.train.batch_size=1 dataloaders.val.batch_size=5 dataloaders.test.batch_size=5 trainer.accumulate_grad_batches=8"

NAME=AGcat-mult-sig_pretrained_D_hinted_l1_ranger_cosannealstart=25e-2_lr=9e-5_bs=8_PS_resnet18_`date +"%m%d_%H%M%S"`; \
python ./submit_job.py -n $NAME --gpumem 16000 \
-pf ../configs/projects/monocular_frontcam_depth_argo.yml \
-po "experiment_name=$NAME" \
-mf ../configs/experiments/argoverse/self_supervised_frontcam.yml \
-mo "model=attention_guiding_concat_mult_sig_upshuffle_resnet18 model.tri_checkpoint_path='/home/clear/fbartocc/depth_project/Depth/weights/ResNet18_MR_selfsup_D.ckpt' losses=hinted_loss losses.HintedMultiViewPhotometricLoss.supervised_method='sparse-l1' dataloaders.train.batch_size=1 dataloaders.val.batch_size=5 dataloaders.test.batch_size=5 trainer.accumulate_grad_batches=8"



##### mish

NAME=AGcat-mult-sig_scratch_mish_photo_ranger_cosannealstart=25e-2_lr=9e-5_bs=8_PS_resnet18_`date +"%m%d_%H%M%S"`; \
python ./submit_job.py -n $NAME --gpumem 16000 \
-pf ../configs/projects/monocular_frontcam_depth_argo.yml \
-po "experiment_name=$NAME" \
-mf ../configs/experiments/argoverse/self_supervised_frontcam.yml \
-mo "model=attention_guiding_concat_mult_sig_upshuffle_resnet18 dataloaders.train.batch_size=1 dataloaders.val.batch_size=5 dataloaders.test.batch_size=5 trainer.accumulate_grad_batches=8 model.depth_net.options.activation=mish"


NAME=AGcat-mult-sig_pretrained_D_mish_photo_ranger_cosannealstart=25e-2_lr=9e-5_bs=8_PS_resnet18_`date +"%m%d_%H%M%S"`; \
python ./submit_job.py -n $NAME --gpumem 16000 \
-pf ../configs/projects/monocular_frontcam_depth_argo.yml \
-po "experiment_name=$NAME" \
-mf ../configs/experiments/argoverse/self_supervised_frontcam.yml \
-mo "model=attention_guiding_concat_mult_sig_upshuffle_resnet18 model.tri_checkpoint_path='/home/clear/fbartocc/depth_project/Depth/weights/ResNet18_MR_selfsup_D.ckpt' dataloaders.train.batch_size=1 dataloaders.val.batch_size=5 dataloaders.test.batch_size=5 trainer.accumulate_grad_batches=8 model.depth_net.options.activation=mish"


# no end blur

NAME=AGcat-mult-sig_pretrained_D_photo_ranger_cosannealstart=25e-2_lr=9e-5_bs=8_PSneb_resnet18_`date +"%m%d_%H%M%S"`; \
python ./submit_job.py -n $NAME --gpumem 16000 \
-pf ../configs/projects/monocular_frontcam_depth_argo.yml \
-po "experiment_name=$NAME" \
-mf ../configs/experiments/argoverse/self_supervised_frontcam.yml \
-mo "model=attention_guiding_concat_mult_sig_upshuffle_resnet18 model.depth_net.options.blur_at_end=False model.tri_checkpoint_path='/home/clear/fbartocc/depth_project/Depth/weights/ResNet18_MR_selfsup_D.ckpt' dataloaders.train.batch_size=1 dataloaders.val.batch_size=5 dataloaders.test.batch_size=5 trainer.accumulate_grad_batches=8"

NAME=AGcat-mult-sig_scratch_photo_ranger_cosannealstart=25e-2_lr=9e-5_bs=8_PSneb_resnet18_`date +"%m%d_%H%M%S"`; \
python ./submit_job.py -n $NAME --gpumem 16000 \
-pf ../configs/projects/monocular_frontcam_depth_argo.yml \
-po "experiment_name=$NAME" \
-mf ../configs/experiments/argoverse/self_supervised_frontcam.yml \
-mo "model=attention_guiding_concat_mult_sig_upshuffle_resnet18 model.depth_net.options.blur_at_end=False dataloaders.train.batch_size=1 dataloaders.val.batch_size=5 dataloaders.test.batch_size=5 trainer.accumulate_grad_batches=8" \
-b -i