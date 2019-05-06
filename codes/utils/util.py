import collections
import pathlib
import random
import subprocess
from functools import reduce
from functools import wraps
from operator import mul
from random import shuffle
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import os

NP_INT_DATATYPE = np.int


def flatten(d, parent_key='', sep='_'):
    # Logic for flatten taken from https://stackoverflow.com/a/6027615/1353861
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def grouped(iterable, n):
    # Modified from https://stackoverflow.com/questions/5389507/iterating-over-every-two-elements-in-a-list/39038787
    return zip(*[iter(iterable)] * n)


# Taken from https://stackoverflow.com/questions/38191855/zero-pad-numpy-array/38192105
def padarray(A, size, const=1):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant', constant_values=const)


def parse_file(file_name):
    '''Method to read the given input file and return an iterable for the lines'''
    with open(file_name, encoding='utf-8') as f:
        for line in f:
            yield line


def get_device_id(device):
    if (device == "cpu"):
        return -1
    elif (device == "gpu"):
        return None
    else:
        return None


def shuffle_list(*ls):
    """Taken from https://stackoverflow.com/questions/23289547/shuffle-two-list-at-once-with-same-order"""
    l = list(zip(*ls))
    shuffle(l)
    return zip(*l)


def chunks(l, n):
    """
    Taken from https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def reverse_dict(_dict):
    return {v: k for k, v in _dict.items()}


def make_dir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def get_device_name(device_type):
    if torch.cuda.is_available() and "cuda" in device_type:
        return device_type
    return "cpu"


def get_current_commit_id():
    command = "git rev-parse HEAD"
    commit_id = subprocess.check_output(command.split()).strip().decode("utf-8")
    return commit_id


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print("function:{} took: {} sec".format(f.__name__, te - ts))
        return result

    return wrap


def show_tensor_as_image(_tensor):
    plt.imshow(_tensor.astype(np.uint8), origin="lower")
    plt.show()


def save_tensor_as_image(_tensor, file_path):
    plt.imsave(file_path, _tensor.astype(np.uint8), origin="lower")


def get_product_of_iterable(_iterable):
    '''Method to get the product of all the enteries in an iterable'''
    return reduce(mul, _iterable, 1)


def log_pdf(x, mu, std):
    '''Method to compute the log pdf for x under a gaussian
    distribution with mean = mu and standard deviation = std
    Taken from: https://chrisorm.github.io/VI-MC-PYT.html'''

    return -0.5 * torch.log(2 * np.pi * std ** 2) - (0.5 * (1 / (std ** 2)) * (x - mu) ** 2)

def running_mean(x, N):
    # Taken from https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)