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
import dnnlib
import torchvision
import scipy.linalg
import cv2
import click
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
from timm import create_model

cache_path = '~/.cache'
_feature_detector_cache = dict()

#load data
rank=0
N=20000000
device = 'cuda'

def choose(real_image_dataset):
    number=0
    files=os.listdir(real_image_dataset)
    files.sort()
    fid_cams=[]
    for f in files:
        fname=os.path.join(real_image_dataset,f)
        try:
            img = PIL.Image.open(fname)
            print(img)
        except(OSError, NameError):
            print('OSError, Path:', fname)
            #os.remove(fname)
            number+=1
    return number

def label_get(real_image_dataset,detector,batch_size=16):
    device = torch.device('cuda', rank)
    if detector == 'inception_v3':
        preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    elif detector == 'resnet50':
        preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    elif detector == 'convnext':
        from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        NORMALIZE_MEAN = IMAGENET_DEFAULT_MEAN
        NORMALIZE_STD = IMAGENET_DEFAULT_STD
        preprocess =transforms.Compose([
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])
    elif 'vitcls' in detector or 'deit' in detector:
        from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        NORMALIZE_MEAN = IMAGENET_DEFAULT_MEAN
        NORMALIZE_STD = IMAGENET_DEFAULT_STD
        preprocess =transforms.Compose([
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ])
    elif 'swin' in detector or 'resmlp' in detector:
        from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        NORMALIZE_MEAN = IMAGENET_DEFAULT_MEAN
        NORMALIZE_STD = IMAGENET_DEFAULT_STD
        preprocess =transforms.Compose([
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ])
    elif 'repvgg' in detector:
        from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        NORMALIZE_MEAN = IMAGENET_DEFAULT_MEAN
        NORMALIZE_STD = IMAGENET_DEFAULT_STD
        preprocess =transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]) 
    
    image_datasets=dataset.ImageFolderDataset(path=real_image_dataset)
    #load detector
    #detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    #detector = get_feature_detector(url=detector_url, device=device, num_gpus=1, rank=rank)
    #detector=torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True).to(device)
    if detector == 'convnext':
        model_name = 'convnext_base'
        detector = create_model(model_name, pretrained=True).to(device)
    elif detector == 'vitcls':
        model_name = 'vit_base_patch16_224'
        detector = create_model(model_name, pretrained=True).to(device)
    elif detector == 'swin':
        model_name = "swin_base_patch4_window7_224"
        detector = create_model(model_name, pretrained=True).to(device)
    elif detector == 'deit':
        model_name = "deit_base_patch16_224"
        detector = create_model(model_name, pretrained=True).to(device)
    elif detector == 'repvgg':
        model_name = "repvgg_b3"
        detector = create_model(model_name, pretrained=True).to(device)
    elif detector == 'resmlp':
        model_name = "resmlp_24_224_dino"
        detector = create_model(model_name, pretrained=True).to(device) 
    else:
        detector=torch.hub.load('pytorch/vision:v0.10.0', detector, pretrained=True).to(device)
    detector.eval()
    device = torch.device('cuda', rank)
    
    image_datasets=dataset.ImageFolderDataset(path=real_image_dataset)
    #load detector
    #detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    #detector = get_feature_detector(url=detector_url, device=device, num_gpus=1, rank=rank)
    #detector=torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True).to(device)
    # detector=torch.hub.load('pytorch/vision:v0.10.0', detector, pretrained=True).to(device)
    # detector.eval()
    # detector=torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True).to(device)
    # detector.eval()
    
    # Read the categories
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    labels={}
    items=0
    for images, _labels in DataLoader(dataset=image_datasets, batch_size=batch_size):
    #detector.layers.mixed_10.register_forward_hook(getActivation("before_pool3"))
    #get features
        new_images=None
        with torch.no_grad():
            #image preprocess
            for image in images:
                new_img = transforms.ToPILImage()(image).convert('RGB')
                image=preprocess(new_img).unsqueeze(0).to(device)
                if new_images is None:
                    new_images=image
                else:
                    new_images=torch.cat((new_images,image),0).to(device)
            
            #forward
            feature_fwd = detector(new_images.to(device))
            items+=batch_size
            items=min(N,items)
            print(f'items : {items}')
            for feature in feature_fwd:
                probability = torch.nn.functional.softmax(feature, dim=0)
                top1_prob, top1_catid = torch.topk(probability, 1)
                label=categories[top1_catid[0]]
                if label in labels:
                    labels[label]+=1
                else:
                    labels[label]=1
    return labels


@click.command()
@click.option('--real_dataset', help='Real dataset to evaluate', type=str, default=None, metavar='[ZIP|DIR]', show_default=True)
@click.option('--gen_dataset', help='Generated dataset to evaluate', type=str, default=None, metavar='[ZIP|DIR]', show_default=True)
@click.option('--detector', help='Choose detector', type=click.Choice(['inception_v3', 'resnet50','convnext', 'vitcls', 'swin', 'repvgg', 'resmlp', 'deit']), default='inception_v3', show_default=True)
@click.option('--histogram_save', help='Where to save the histogram', type=str, default=None, metavar='STRING', show_default=True)

def label_match(
    real_dataset: str,
    gen_dataset: str,
    detector: str,
    histogram_save: str,
    ):
    
    real_dataset_url=real_dataset
    gen_dataset_url=gen_dataset
    real_labels=label_get(real_dataset_url,detector,batch_size=1000)
    gen_labels=label_get(gen_dataset_url,detector,batch_size=1000)

    index = np.arange(30)
    bar_width = 0.35
    p_real=dict(sorted(real_labels.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)[:30])
    x_label=list(p_real.keys())
    y_real=p_real.values()
    y_gen=[]
    for x in x_label:
        y_gen.append(gen_labels[x])
    
    bar1=plt.bar(index, y_real, bar_width, label='real dataset')
    bar2=plt.bar(index+bar_width, y_gen, bar_width, color='orange', label='gen dataset')
    plt.xticks(index + bar_width, x_label,rotation=90)
    plt.title('The match figure')
    plt.legend()
    plt.savefig(histogram_save,bbox_inches = 'tight',dpi=300)


if __name__ == "__main__":
    label_match()
    
    
    
