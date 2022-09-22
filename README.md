


<div align="center">   
 
# LiDARTouch: Monocular metric depth estimation with a few-beam LiDAR

<!---
[Setup](#setup) // [Vizualisation](#vizualisation) // [Weights](#weights) // [Data-preparation](#data) //
[Training](#training) // [License](#license) // [Acknowledgements](#acknowledgements)
--->

<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.2109.03569-B31B1B.svg)](https://arxiv.org/abs/2109.03569)


This is the reference PyTorch implementation for training and testing depth prediction models using the method described 
in our paper [**LiDARTouch: Monocular metric depth estimation with a few-beam LiDAR**
](https://arxiv.org/abs/2109.03569)
</div>


If you find our work useful, please consider citing:
```bibtex
@misc{bartoccioni2021lidartouch,
    title={LiDARTouch: Monocular metric depth estimation with a few-beam LiDAR},
    author={Florent Bartoccioni and Éloi Zablocki and Patrick Pérez and Matthieu Cord and Karteek Alahari},
    year={2021},
    eprint={2109.03569},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## ⚙ Setup <a name="setup"></a>

### Environment
First, clone the repo
```bash
# clone project   
git clone https://github.com/F-Barto/LiDARTouch
cd LiDARTouch
```

Then, create the [conda](https://docs.conda.io/en/latest/miniconda.html) environment,
install dependencies and activate env.
```bash
# create conda env and install dependancies 
conda env create -n LiDARTouch -f environment.yaml
conda activate LiDARTouch
pip install -e .
 ```

<!---
## 🤩 Vizualisation <a name="vizualisation"></a>


## 💪 Pre-trained weights <a name="weights"></a>
--->

## 💾 Data-preparation <a name="data"></a>
To train the model from scratch on KITTI you first need to download both:
- the [raw data](http://www.cvlibs.net/datasets/kitti/raw_data.php)
- the [depth completion data](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion)

⚠️the data weighs about 200GB

Once the data downloaded you need to preprocess it. 
ℹ️Note that we provide the data split files under `data_splits`

Under the `scripts/kitti_data_preparation` folder you will find:
- `lidar_sparsification.py`
- `prepare_split_data.py`

### Step 1: LiDAR sparsification
This script virtually sparsify the raw 64-beam LiDAR to a 4-beam LiDAR; use as follows:
```
python lidar_sparsification.py KITTI_RAW_ROOT_DIR OUTPUT_DIR DATA_SPLIT_DIR SPLIT_FILE_NAMES [OPTIONS]
```
e.g.,
```
python ./lidar_sparsification.py \
/path_to_kitti_root_folder/KITTI_raw/  \
/path_to_sparfied_lidar_data/sparsified_lidar/ \
/path_to_LiDARTouch_folder/LiDARTouch/data_splits \
'eigen_train_files.txt,filtered_eigen_val_files.txt,filtered_eigen_test_files.txt' \
--downsample_factor=16
```
the parameter `--downsample_factor=16` indicates that only 1 out of 16 beams will be kept (leading to 4 beam).
Alternatively, you can choose to select individual beams by their indexes with `--downsample_indexes='5,7,9,11,20`.

### Step 2: Temporal context and Pose pre-computation
Then we will create a pickle `split_data` containing the data for:
- the source views available for each image listed in the split file
- the relative pose between the source and target views using the IMU and/or Perspective-n-point w/ LiDAR

This script is used as follows:
```
prepare_split_data.py KITTI_RAW_ROOT_DIR OUTPUT_PATH DATA_SPLIT_DIR SPLIT_FILE_NAMES SOURCE_VIEWS_INDEXES [OPTIONS]
```
e.g.,
```
python ./prepare_split_data.py \
/path_to_kitti_root_folder/KITTI_raw/  \
/path_to_output/split_data.pkl \
/path_to_LiDARTouch_folder/LiDARTouch/data_splits \
'eigen_train_files.txt,filtered_eigen_val_files.txt,filtered_eigen_test_files.txt' \
'[-1,1]' \
--imu \
--pnp /path_to_sparfied_lidar_data/sparsified_lidar/factor_16
```
use `--help` for more details.

### Paths configuration

Change the paths present in the `.env` file to configure the saving dir and the path to your dataset.

## 🏋️ Training <a name="training"></a>

Monodepth2 depth network only photometric supervision (relative depth | infinite depth issue)
```
python train.py experiment=PoseNet_P_multiscale depth_net=monodepth2
```

Monodepth2 depth network with IMU supervision (metric depth | infinite depth issue)
```
python train.py experiment=PoseNet_P+IMU_multiscale depth_net=monodepth2
```

Monodepth2-L depth network with LiDARTouch supervision (metric depth | NO infinite depth issue)
```
python train.py experiment=PnP_P+ml1L4_multiscale depth_net=monodepth2lidar
```

If you would like to use other neural network architectures please refer to [**TODO**].

Regarding the infinite depth problem, the two major factors alleviating it are the auto-masking and the LiDAR self-supervision.
In practice, we found multi-scale supervision and the smoothness loss to be critical for stable training when using the LiDAR self-supervision.

<!---
## 👩‍⚖️License <a name="license"></a>
--->

## 🎖️ Acknowledgements <a name="acknowledgements"></a>

This work and code base is based upon the papers and code base of:
- [PackNet-sfm](https://github.com/TRI-ML/packnet-sfm)
- [Monodepth2](https://github.com/nianticlabs/monodepth2)
- [Selfsup Sparse-to-Dense](https://github.com/fangchangma/self-supervised-depth-completion)

In particular, to structure our code we used:
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

Please consider giving these projects a star or citing their work if you use them.



