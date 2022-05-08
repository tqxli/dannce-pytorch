![Image](./common/dannce_logo.png)

## Overview
This repository contains the **PyTorch** implementation of [**DANNCE**](https://github.com/spoonsso/dannce/tree/master) (3-Dimensional Aligned Neural Network for Computational Ethology) for 3D animal pose estimation.

![Image](./common/Figure1.png)

### Important notice
This release is ahead the stable Tensorflow version and is consistently under active development. 


## DANNCE Installation

1. Clone the github repository: \
```
git clone --recursive https://github.com/tqxli/dannce-pytorch.git`
cd dannce
```
2. If you do not already have it, install [Anaconda](https://www.anaconda.com/products/individual).

3. Set up a new Anaconda environment with the following configuration: \
`conda create -n dannce python=3.7`

4. Activate the new Anaconda environment: \
`conda activate dannce`

5. Install PyTorch: \
`conda install pytorch==1.9.0 torchvision==0.10.1 cudatoolkit=11.1 -c pytorch -c conda-forge`

6. Update setuptools: \
`pip install -U setuptools`

7. Install DANNCE with the included setup script from within the base repository directory: \
`pip install -e .`