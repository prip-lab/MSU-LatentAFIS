# loaders.py

import torch
import numpy as np
import scipy.io as sio
from skimage import io
from PIL import Image


def loader_skimage(path):
    return io.imread(path)


def loader_image(path):
    return Image.open(path).convert('RGB')


def loader_torch(path):
    return torch.load(path)


def loader_numpy(path):
    return np.load(path)


def loader_mat(path):
    return sio.loadmat(path)
