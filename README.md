# Depth Completion through Colorization

The algorithm has been tested on WSL2 Ubuntu 20.04 using Python 3.7.

To get necessary dependencies, you may use
```
pip install numpy scipy pillow gdown numba opencv-python
```

## Test for solid-state LiDAR dataset

To setup the dataset, you may use
```
mkdir data
bash ./bash/setup_dataset.sh
```

For depth completion, change the parameters in ./bash/colorization_ssl.sh, you may use
```
bash ./bash/colorization_ssl.sh
```

## Test for KITTI depth completion validation dataset

To setup the KITTI dataset, you may use
```
mkdir data
bash ./bash/setup_dataset_kitti.sh
```

For depth completion, you may use
```
bash ./bash/colorization_kitti.sh
```

To test for colorization with guidance image, change the path for your guidance image. The example use the prediction from colorization with equal weights as guidance, you may use
```
bash ./bash/colorization_depth_guidance.sh
```

The code for the algorithm and evaluation is adapted from open source github repository.