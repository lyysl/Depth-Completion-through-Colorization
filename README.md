# Depth Completion through Colorization

The algorithm has been tested on WSL2 Ubuntu 20.04 using Python 3.7.

To get necessary dependencies, you may use
```
pip install numpy scipy pillow gdown numba
```

To setup the dataset, you may use
```
mkdir data
bash ./bash/setup_dataset.sh
```

For depth completion, change the parameters in ./bash/colorization_ssl.sh, you may use
```
bash ./bash/colorization_ssl.sh
```

The code for the algorithm and evaluation is adapted from open source github repository.