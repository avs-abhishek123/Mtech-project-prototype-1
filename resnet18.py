import os
# loads the OS 

import time
#uses time

from PIL import Image
# use to manipulate images

#from pdb import set_trace as bp 
# can be used for putting up a break point while debugging

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt

import numpy as np
# used to manipulate arrays

import torch
# Deep Learning applications using GPUs and CPUs.

from torch.utils import data
from torch import nn
# module to help us in creating and training of the neural network

from torch import optim
# package implementing various optimization algorithms

import torch.nn.functional as F
# Used to apply Convolution functions etc
# https://pytorch.org/docs/stable/nn.functional.html

from torchvision import datasets, transforms, models

import sys