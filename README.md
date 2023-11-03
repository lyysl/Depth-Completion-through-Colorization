# Depth Completion through Colorization

The algorithm has been tested on WSL2 Ubuntu 20.04 using Python 3.7.

To get necessary dependencies, you may use
```
pip install numpy scipy pillow gdown
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

The algorithm is adapted from [[fill_depth_colorization.py]](https://gist.github.com/ialhashim/be6235489a9c43c6d240e8331836586a)

The evaluation is adapted from [[learning-topology-synthetic-data]](https://github.com/alexklwong/learning-topology-synthetic-data)