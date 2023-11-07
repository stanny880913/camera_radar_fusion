#!/bin/bash

# Install PyTorch and related packages
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge

# Install mmcv-full
pip install mmcv-full==1.3.12 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# Install mmdet
pip install mmdet==2.14.0

# Install mmsegmentation
pip install mmsegmentation==0.14.1

# Uninstall and reinstall numpy
pip uninstall numpy -y
pip install numpy==1.19.5

# Install mmpycocotools
pip install mmpycocotools

# Install pycocotools
pip install pycocotools==2.0.1

# Clone the mmdetection3d repository and install it
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.16.0
pip install -v -e .

# Install opencv-python-headless
pip install opencv-python-headless

# Uninstall and reinstall yapf
pip uninstall yapf -y
pip install yapf==0.40.1
