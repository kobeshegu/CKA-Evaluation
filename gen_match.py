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
import pandas as pd
from tkinter import W
import uuid
import numpy as np
import torch
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
from typing import List, Optional, Tuple, Union
import click
import legacy
from timm import create_model


#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

def pre_process_func(model):
    if model == 'inception':
        preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    elif model == 'resnet50':
        preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    elif model == 'convnext':
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
    elif model == 'vitcls':
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
    return preprocess

def label_get(new_generated_image, preprocess, detector):
    rank=0
    N=50000
    device = torch.device('cuda', rank)
    #detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    #image=cv2.imread(new_generated_image)
    #back_image=cv2.cvtColor(np.array(image),cv2.COLOR_BGR2RGB)
    #input_image=torch.from_numpy(back_image.transpose((2,0,1))).unsqueeze(0).to(device)
    image=new_generated_image
    #transform = torchvision.transforms.Compose([transforms.Resize(299), transforms.CenterCrop(299), torchvision.transforms.ToTensor()])
    #image = transform(image).unsqueeze(0).to(device) * 255
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        detector.to('cuda')
    #detector.layers.mixed_10.register_forward_hook(getActivation("before_pool3"))
    #get features
    feature_fwd = detector(input_batch).requires_grad_(True)
    probabilities = torch.nn.functional.softmax(feature_fwd[0], dim=0)
    
    # Read the categories
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    top1_prob, top1_catid = torch.topk(probabilities, 1)
    label = categories[top1_catid[0]]
    print(label)
    return label

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--limit', type=float, help='domain of walker', default=0.02, show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--detector', help='Choose detector', type=click.Choice(['inception', 'resnet50', 'convnext', 'vitcls']), default='inception', show_default=True)
@click.option('--translate', help='Translate XY-coordinate (e.g. \'0.3,1\')', type=parse_vec2, default='0,0', show_default=True, metavar='VEC2')
@click.option('--cfg', help='Base configuration', type=click.Choice(['stylegan3', 'stylegan2']), required=True)
@click.option('--rotate', help='Rotation angle in degrees', type=float, default=0, show_default=True, metavar='ANGLE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--inception_label', help='Where to read the label of dateset matched by inception', type=str, default=None, metavar='STRING', show_default=True)
@click.option('--resnet50_label', help='Where to read the label of dateset matched by resnet50', type=str, default=None, metavar='STRING', show_default=True)
@click.option('--convnext_label', help='Where to read the label of dateset matched by convnext', type=str, default=None, metavar='STRING', show_default=True)
@click.option('--vitcls_label', help='Where to read the label of dateset matched by supervised vit', type=str, default=None, metavar='STRING', show_default=True)
@click.option('--gen_inception_label', help='Where to read the label of gen_dateset matched by inception', type=str, default=None, metavar='STRING', show_default=True)
@click.option('--gen_resnet50_label', help='Where to read the label of gen_dateset matched by resnet50', type=str, default=None, metavar='STRING', show_default=True)
@click.option('--gen_convnext_label', help='Where to read the label of gen_dateset matched by convnext', type=str, default=None, metavar='STRING', show_default=True)
@click.option('--gen_vitcls_label', help='Where to read the label of gen_dateset matched by supervised vit', type=str, default=None, metavar='STRING', show_default=True)
@click.option('--seed_save', help='Where to save the seeds chosen', type=str, default=None, metavar='STRING', show_default=True)
@click.option('--histogram_save', help='Where to save the histogram', type=str, default=None, metavar='STRING', show_default=True)
@click.option('--num_real', help='Number of real dataset', type=int, default=50000, metavar='INT', show_default=True)
def generate_images(
    network_pkl: str,
    seeds: List[int],
    limit: float,
    truncation_psi: float,
    noise_mode: str,
    detector: str,
    cfg: str,
    outdir: str,
    inception_label: str,
    resnet50_label: str,
    convnext_label: str,
    vitcls_label: str,
    gen_inception_label: str,
    gen_resnet50_label: str,
    gen_convnext_label: str,
    gen_vitcls_label: str,
    seed_save: str,
    histogram_save: str,
    num_real: int,
    translate: Tuple[float,float],
    rotate: float,
    class_idx: List[int]
    
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained AFHQv2 model ("Ours" in Figure 1, left).
    python gen_images.py --outdir=out --trunc=1 --seeds=2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

    \b
    # Generate uncurated images with truncation using the MetFaces-U dataset
    python gen_images.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfacesu-1024x1024.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(cfg, f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Labels.
    class_label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')
    filename=None
    if detector == 'inception':
        filename = inception_label
    elif detector == 'resnet50':
        filename = resnet50_label
    elif detector == 'convnext':
        filename = convnext_label
    elif detector =='vitcls':
        filename = vitcls_label
    if filename is not None:
        with open(filename,'rb') as fo:
            labels=pickle.load(fo,encoding='bytes')
            fo.close()
        origin_labels=labels
    N=50000
    full_number=0
    # Generate images.
    gen_labels={}
    if seed_save is not None:
        f=open('matched_seed.txt','w')
    if detector == 'inception':
        gen_filename = gen_inception_label
    elif detector == 'resnet50':
        gen_filename = gen_resnet50_label
    elif detector == 'convnext':
        gen_filename = gen_convnext_label
    elif detector == 'vitcls':
        gen_filename = gen_vitcls_label
    if gen_filename is not None:
        with open(gen_filename,'rb') as fg:
            gen_label_origin=pickle.load(fg,encoding='bytes')
            fg.close()
    if filename is not None:
        for label in labels:
            gen_labels[label]=0
            if gen_filename is not None and label in gen_label_origin:
                gen_labels[label]+=gen_label_origin[label]
    pre_process = pre_process_func(model=detector)
    #load detector
    #detector = get_feature_detector(url=detector_url, device=device, num_gpus=1, rank=rank)
    if detector == 'resnet50':
        detector=torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).to(device)
    if detector == 'inception':
        detector=torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True).to(device)
    if detector == 'convnext':
        model_name = 'convnext_base'
        detector = create_model(model_name, pretrained=True).to(device)
    if detector == 'vitcls':
        model_name = 'vit_base_patch16_224'
        detector = create_model(model_name, pretrained=True).to(device)
    detector.eval()
    for seed_idx, seed in enumerate(seeds):
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

        # Construct an inverse rotation/translation matrix and pass to the generator.  The
        # generator expects this matrix as an inverse to avoid potentially failing numerical
        # operations in the network.
        if hasattr(G.synthesis, 'input'):
            m = make_transform(translate, rotate)
            m = np.linalg.inv(m)
            G.synthesis.input.transform.copy_(torch.from_numpy(m))
        img = G(z, class_label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        image=Image.fromarray(img[0].cpu().numpy(), 'RGB')
        gen_label=label_get(image, pre_process, detector)
        if filename is None:
            if gen_label not in gen_labels:
                gen_labels[gen_label]=0
            
            if gen_labels[gen_label]>=50:
                print('label limit!'+': '+gen_label)
            elif gen_labels[gen_label]<50 and full_number<N:
                gen_labels[gen_label]+=1
                print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
                full_number+=1
                image.save(f'{outdir}/seed{seed:04d}.png')
                seed_name=f'{seed:04d}\n'
                if seed_save is not None:
                    f.write(seed_name)
                if full_number==N:
                    break
        else:
            if gen_label in labels:
                #if gen_labels[gen_label]>=1:
                if float(gen_labels[gen_label])*float(num_real/50000)>float(origin_labels[gen_label]*(1+limit)):
                    print('label limit!')
                print(float(origin_labels[gen_label]*(1+limit)),float(gen_labels[gen_label])*float(num_real/50000))
                #if gen_labels[gen_label]<1 and full_number<N:
                if float(gen_labels[gen_label])*float(num_real/50000)<=float(origin_labels[gen_label]*(1+limit)) and full_number<N:
                    gen_labels[gen_label]+=1
                    print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
                    full_number+=1
                    image.save(f'{outdir}/seed{seed:04d}.png')
                    seed_name=f'{seed:04d}\n'
                    if seed_save is not None:
                        f.write(seed_name)
                    if full_number==N:
                        break
    if seed_save is not None:
        f.close()
    if histogram_save is not None:
        index = np.arange(20)
        bar_width = 0.35
        p_real=dict(sorted(origin_labels.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)[:20])
        x_label=list(p_real.keys())
        y_real=p_real.values()
        y_gen=[]
        for x in x_label:
            y_gen.append(gen_labels[x])
    
        bar1=plt.bar(index, y_real, bar_width, label='real dataset')
        bar2=plt.bar(index+bar_width, y_gen, bar_width, color='orange', label='gen_match dataset')
        plt.xticks(index + bar_width, x_label,rotation=90)
        plt.title('The match figure')
        plt.legend()
        plt.savefig(histogram_save)
        
        
        
        
#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
