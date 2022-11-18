![Image](./common/dannce_logo.png)

## News
- 2022/08: Major codebase refactoring, full features converted to PyTorch. Notice this is not a stable release; please refer to the original Tensorflow version if necessary.
- 2022/06: Check out our latest paper presented at CVPR 2022 CV4Animals Workshop
 
   [**Improved Markerless 3D Animal Pose Estimation Using Temporal Semi-Supervision**](https://drive.google.com/file/d/1lZcrXlazErthSkPJtR6g1DLhFo1HmwXU/view?usp=sharing) Tianqing Li, Kyle S. Severson, Fan Wang and Timothy W. Dunn, 2022. 

   Please check out the branch `official_temporal_release` for more details.
- 2021/04: Our paper [**Geometric deep learning enables 3D kinematic profiling across species and environments**](https://www.nature.com/articles/s41592-021-01106-6) is in press in Nature Methods.

## Overview
This repository contains the **PyTorch** implementation of [**DANNCE**](https://github.com/spoonsso/dannce/tree/master) (3-Dimensional Aligned Neural Network for Computational Ethology) for 3D animal pose estimation. Compared to existing approaches for 2D keypoint detection in animals, DANNCE leverages 3D volumetric representations sharing similar inspirations with [Learnable Triangulation of Human Pose, ICCV 2019](https://arxiv.org/abs/1905.05754) and [VoxelPose, ECCV 2020](https://arxiv.org/abs/2004.06239) and directly infers keypoint positions in 3D space. 

![Image](./common/Figure1.png)

## Installation
1. Clone this github repository
```
git clone --recursive https://github.com/tqxli/dannce-pytorch.git
cd dannce-pytorch
```

2. If you do not already have it, install [Anaconda](https://www.anaconda.com/products/individual).

3. Set up a new Anaconda environment with the following configuration: \
`conda create -n dannce_pytorch python=3.7 cudatoolkit=11.1 cudnn ffmpeg -c nvidia`

4. Activate the new Anaconda environment: \
`conda activate dannce_pytorch`

5. Install PyTorch: \
`conda install pytorch=1.9.0 torchvision=0.10.0 -c pytorch`

6. Update setuptools: \
`pip install setuptools==59.5.0`

7. Install with the included setup script: \
`pip install -e .`

## Running demo
To test your DANNCE installation and familiarize yourself with DANNCE file and configuration formatting, run DANNCE predictions for `markerless_mouse_1`. Because the videos and checkpoints are too large to host on GitHub, run the following commands from the base `dannce` repository to download necessary files and place them in each associated location:

For markerless_mouse_1: 
```
wget -O vids.zip https://tinyurl.com/DANNCEmm1vids; 
unzip vids.zip -d vids; 
mv vids/* demo/markerless_mouse_1/videos/; 
rm -r vids vids.zip; 
```

For markerless_mouse_2: 
```
wget -O vids2.zip https://tinyurl.com/DANNCEmm2vids; 
unzip vids2.zip -d vids2; 
mv vids2/* demo/markerless_mouse_2/videos/; 
rm -r vids2 vids2.zip 
``` 

For the checkpoint, download from [this link](https://duke.box.com/s/tlpw8phcf09f0oqh5m2wa6vnvuwxb2gi) and place in `demo/markerless_mouse_1/DANNCE/weights`.

Once the files are downloaded and placed, run: 
```
cd demo/markerless_mouse_1/; 
dannce-predict ../../configs/dannce_mouse_config.yaml
```

This demo will inference over 1000 frames of mouse data and save the results to: \
```
demo/markerless_mouse_1/DANNCE/predict_results/save_data_AVG0.mat
```

To train a new model FROM SCRATCH with the demo data, run
```
dannce-train ../../configs/dannce_mouse_config.yaml
```

To finetune a previous checkpoint, run
```
dannce-train ../../configs/dannce_mouse_config.yaml --train-mode finetune --epochs 100
```

To work with a larger dataset, you may turn on `--use-npy True` to pre-generate the volumes to disk.