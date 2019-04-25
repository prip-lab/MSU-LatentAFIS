# utils.py

import os
import csv
import math
import numpy as np
import argparse
from inspect import getframeinfo, stack
import json
import sys
import psutil
import signal


def setup_graceful_exit():
    # handle Ctrl-C signal
    signal.signal(signal.SIGINT, ctrl_c_handler)


def cleanup():
    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    for child in children:
        try:
            os.kill(int(child.pid), signal.SIGKILL)
        except OSError as ex:
            raise Exception("wasn't able to kill the child process (pid:{}).".format(child.pid))
    sys.exit(0)


def ctrl_c_handler(*kargs):
    cleanup()


def isnan(x):
    return x != x


def _debuginfo(self, *message):
    """Prints the current filename and line number in addition to debugging
    messages."""
    caller = getframeinfo(stack()[1][0])
    print('\033[92m', caller.filename, '\033[0m', caller.lineno,
          '\033[95m', self.__class__.__name__, '\033[94m', message, '\033[0m')


def readcsvfile(filename, delimiter):
    with open(filename, 'r') as f:
        content = []
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            content.append(row)
    f.close()
    return content


def readtextfile(filename):
    with open(filename) as f:
        content = f.readlines()
    f.close()
    return content


def writetextfile(data, filename, path=None):
    """If path is provided, it will make sure the path exists before writing
    the file."""
    if path:
        if not os.path.isdir(path):
            os.makedirs(path)
        filename = os.path.join(path, filename)
    with open(filename, 'w') as f:
        f.writelines(data)
    f.close()


def delete_file(filename):
    if os.path.isfile(filename) is True:
        os.remove(filename)


def eformat(f, prec, exp_digits):
    s = "%.*e" % (prec, f)
    mantissa, exp = s.split('e')
    # add 1 to digits as 1 is taken by sign +/-
    return "%se%+0*d" % (mantissa, exp_digits + 1, int(exp))


def file_exists(filename):
    return os.path.isfile(filename)


def str2bool(v):
    """A Parser for boolean values with argparse"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def gaussian(size, center, sigma=1):
    if np.isnan(center[0]) or np.isnan(center[1]):
        return np.zeros(size)

    x, y = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    den = 2 * pow(sigma, 2)
    num = np.power(x - x0, 2) + np.power(y - y0, 2)
    return np.exp(-(num / den)) / math.sqrt(2 * np.pi * sigma * sigma)


def plotlify(fig, env='main', win='mywin'):
    fig = {key: fig[key] for key in fig.keys()}
    fig['win'] = win
    fig['eid'] = env

    return fig

def normalize(x, tool, axis=None, epsilon=10e-12):
    ''' Devide the vectors in x by their norms.'''
    if axis is None:
        axis = len(x.shape) - 1
    if tool == 'numpy':
        norm = np.linalg.norm(x, axis=axis, keepdims=True)
    elif tool == 'torch':
        norm = torch.mul(x,x).sum(dim=axis, keepdim=True).sqrt()
    x = x / (norm + epsilon)
    return x
