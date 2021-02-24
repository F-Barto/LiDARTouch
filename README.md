<div align="center">   
 
# Depth  

[![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)](https://www.python.org/downloads/release/python-370/)
[![Pytorch Lightning](https://img.shields.io/badge/pytorch%20lightning-7.1.0-blueviolet.svg)]()

**Note: this is a repo for personnal use. I don't have a ssh connection I have to push pull on remote server at every 
little modification; hence the dirty commits. The doc is not complete and may even be erroneous. In addition, what works 
today may not works tomorrow. Lastly, the scripts undar ``tasks/`` are made for the INRIA's cluster.**

</div>
 
## Description


## Repo organisation

* data-preparation
* networks
* models
* train.py

# How to run   

## Installation
First, clone the repo
```bash
# clone project   
git clone https://github.com/F-Barto/Depth
cd Depth
```

Then, install dependencies and activate env.
```bash
# create conda env and install dependancies 
conda env create -n Depth_env -f environment.yml
conda activate Depth_env
 ```  

### ACMNet

Require CUDA 10.0 installed and add to .bashrc

```
export CUDA_ROOT=/path/to/cuda/cuda-10.0
export LD_LIBRARY_PATH=$CUDA_ROOT/lib64
export CUDA_HOME=$CUDA_ROOT
export LIBRARY_PATH=$LD_LIBRARY_PATH
export PATH=$CUDA_ROOT/bin:$PATH
```

```
conda env create -n ACMNet -f environment_ACMNet.yml
conda activate ACMNet
```

then install pointlib
```
git pull ACMNet somewhere
cd [...]/ACMNet
pip install pointlib/.
cd [...]/Depth
```

---

if need to reinstall, remove env and clean cache
```
conda remove --name ACMNet --all
conda clean --all --yes
```
clean `pointlib/` folder
```
rm pointlib.cpython-36m-x86_64-linux-gnu.so
rm pointlib.cpython-38-x86_64-linux-gnu.so
rm -rf build
rm -rf pointlib.egg-info/
```

## Data-preparation

DL Kitti
DL Argo
Install argo api

## Running the code:

## Generated output files/dirs

# List of Depth Estimation Methods Implemented
Please cite the methods below if you use them.

If you want to load weights from PackNet you have to also install yacs and then:
>>> import torch
>>> path = "./ResNet18_MR_selfsup_K.ckpt"
>>> ckpt = torch.load(path, map_location='cpu')
>>> list(ckpt.keys())
['config', 'state_dict']
>>> list(ckpt['state_dict'].keys())
['model.depth_net.encoder.encoder.conv1.weight', 'model.depth_net.encoder.encoder.bn1.weight', ...,
 'model.pose_net.decoder.net.3.weight', 'model.pose_net.decoder.net.3.bias']
