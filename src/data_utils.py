'''
Authors: Alex Wong <alexw@cs.ucla.edu>, Safa Cicek <safacicek@ucla.edu>

If this code is useful to you, please cite the following paper:
A. Wong, S. Cicek, and S. Soatto. Learning topology from synthetic data for unsupervised depth completion.
In the Robotics and Automation Letters (RA-L) 2021 and Proceedings of International Conference on Robotics and Automation (ICRA) 2021

@article{wong2021learning,
    title={Learning topology from synthetic data for unsupervised depth completion},
    author={Wong, Alex and Cicek, Safa and Soatto, Stefano},
    journal={IEEE Robotics and Automation Letters},
    volume={6},
    number={2},
    pages={1495--1502},
    year={2021},
    publisher={IEEE}
}
'''
import numpy as np
from PIL import Image


def load_depth_with_validity_map(path, multiplier=256.0):
    '''
    Loads a depth map from a 16/32-bit PNG file

    Args:
        path : str
            path to 16/32-bit PNG file
        multiplier : float
            depth factor multiplier for saving and loading in 16/32 bit png
    Returns:
        numpy : depth map
        numpy : binary validity map for available depth measurement locations
    '''

    # Loads depth map from 16/32-bit PNG file
    z = np.array(Image.open(path), dtype=np.float32)

    # Assert 16/32-bit (not 8-bit) depth map
    z = z / multiplier
    z[z <= 0] = 0.0
    v = z.astype(np.float32)
    v[z > 0]  = 1.0
    return z, v

def load_depth(path, multiplier=256.0):
    '''
    Loads a depth map from a 16/32-bit PNG file

    Args:
        path : str
            path to 16/32-bit PNG file
        multiplier : float
            depth factor multiplier for saving and loading in 16/32 bit png
    Returns:
        numpy : depth map
    '''

    # Loads depth map from 16/32-bit PNG file
    z = np.array(Image.open(path), dtype=np.float32)

    # Assert 16/32-bit (not 8-bit) depth map
    z = z / multiplier
    z[z <= 0] = 0.0
    return z


def pad_batch(filepaths, n_batch):
    '''
    Pads the filepaths based on the batch size (n_batch)
    e.g. if n_batch is 8 and number of filepaths is 14, then we pad with 2

    Args:
        filepaths : list
            list of filepaths to be read
        n_batch : int
            number of examples in a batch
    Returns:
        list : list of paths with padding
    '''

    n_samples = len(filepaths)
    if n_samples % n_batch > 0:
        n_pad = n_batch - (n_samples % n_batch)
        filepaths.extend([filepaths[-1]] * n_pad)

    return filepaths