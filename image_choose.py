from genericpath import isfile
from importlib.resources import path
from inspect import getattr_static
import os
from pyexpat import features
import time
import re
import hashlib
import pickle
import copy
from turtle import color
import pandas as pd
from tkinter import W
import uuid
import numpy as np
import torch
import click
import dnnlib
import torchvision
import scipy.linalg
import cv2
import pickle
from training import dataset
import torch.nn.functional as F
from torch.utils import data
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms,datasets
from torchsummary import summary
from sqrtm import sqrtm
from visualizer import *
from torchvision.io import read_image
from torch.utils.data import DataLoader

cache_path = '~/.cache'
_feature_detector_cache = dict()

#load data
rank=0
N=20000000
device = 'cuda'

def choose(real_image_dataset):
    number=0
    files=os.listdir(real_image_dataset)
    fid_cams=[]
    for f in files:
        fname=os.path.join(real_image_dataset,f)
        try:
            img = Image.open(fname)
        except:
            print('ReadError, Path:', fname)
            #os.remove(fname)
            number+=1
    return number

@click.command()
@click.option('--real_dataset', help='Generated dataset to evaluate', type=str, default=None, metavar='[ZIP|DIR]', show_default=True)
def image_choose(
    real_dataset: str,
    
):
    
    real_dataset_url=real_dataset
    res=choose(real_dataset_url)
    print(res)

if __name__ == "__main__":
    image_choose()
    
    
    
