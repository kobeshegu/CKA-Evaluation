# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Miscellaneous utilities used internally by the quality metrics."""

from concurrent.futures import process
from genericpath import isfile
from importlib.resources import path
from inspect import getattr_static
from mimetypes import init
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
import math
import numpy as np
import random
import torch
import dnnlib
import torchvision
import scipy.linalg
import cv2
import pickle
from mocov3 import vits
from training import dataset
import torch.nn.functional as F
from torch.utils import data
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms,datasets
import torchvision.models as models
from torchsummary import summary
from sqrtm import sqrtm
from visualizer import *
from torchvision.io import read_image
from torch.utils.data import DataLoader
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
#from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights, maskrcc_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from timm import create_model


#----------------------------------------------------------------------------

class MetricOptions:
    def __init__(self, G=None, G_kwargs={}, dataset_kwargs={}, generate_dataset_kwargs={}, num_gpus=1, rank=0, device=None, progress=None, cache=0, feature_save_flag=0, generate=None, feature_network=None, eval_bs=1, resolution=256, 
    layers=None, random=False, random_size=5000, random_num=5, cfg=None, dimension=None, kernel=None, sigma=None, save_res=None, feature_save=None, save_stats=None, num_gen=None,  save_name=None, max_real=None, 
    token_vit=None, cka_normalize=None, post_process=None, subset_estimate=None, num_subsets=None, max_subset_size=None, groups=None, group_index=None, fusion_ways=None, fusion_softmax_order=None, random_projection=None, fuse_all=None, 
    fuse_all_ways=None, detector1=None, detector2=None, save_real_features=None):
        assert 0 <= rank < num_gpus
        self.G              = G
        self.G_kwargs       = dnnlib.EasyDict(G_kwargs)
        self.dataset_kwargs = dnnlib.EasyDict(dataset_kwargs)
        self.generate_dataset_kwargs = dnnlib.EasyDict(generate_dataset_kwargs)
        self.num_gpus       = num_gpus
        self.rank           = rank
        self.device         = device if device is not None else torch.device('cuda', rank)
        self.progress       = progress.sub() if progress is not None and rank == 0 else ProgressMonitor()
        self.cache          = cache
        self.feature_save_flag = feature_save_flag
        self.feature_network = feature_network
        self.eval_bs = eval_bs
        self.resolution = resolution
        self.layers = layers
        self.random = random
        self.random_size = random_size
        self.random_num = random_num
        self.cfg = cfg
        self.generate = generate
        self.dimension = dimension
        self.kernel = kernel
        self.sigma = sigma
        self.save_res = save_res
        self.feature_save = feature_save
        self.save_stats = save_stats
        self.num_gen = num_gen
        self.save_name = save_name
        self.max_real = max_real
        self.token_vit = token_vit
        self.cka_normalize = cka_normalize
        self.post_process = post_process
        self.subset_estimate = subset_estimate
        self.num_subsets =num_subsets
        self.max_subset_size = max_subset_size
        self.groups = groups
        self.group_index = group_index
        self.fusion_ways = fusion_ways
        self.fusion_softmax_order = fusion_softmax_order
        self.random_projection = random_projection
        self.fuse_all = fuse_all
        self.fuse_all_ways = fuse_all_ways
        self.detector1 = detector1
        self.detector2 = detector2
        self.save_real_features = save_real_features
#----------------------------------------------------------------------------

cache_path = '~/.cache/'
_feature_detector_cache = dict()
torchvision_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

model_names = ['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base'] + torchvision_model_names


def get_feature_detector_name(url):
    return os.path.splitext(url.split('/')[-1])[0]

def get_feature_detector(url, opts, device=torch.device('cpu'), num_gpus=1, rank=0, verbose=False):
    assert 0 <= rank < num_gpus
    key = (url, device)
    is_leader = (rank == 0)
    if not is_leader and num_gpus > 1:
        torch.distributed.barrier() # leader goes first
    if url is not None:
        fn = os.path.split(url)[-1]
    if 'swav' in opts.feature_network:
        from training.resnet_swav import resnet50
        detector = resnet50(url=url).eval().requires_grad_(False).to(device)
        _feature_detector_cache[key] = detector
    elif 'clip' in opts.feature_network:
        from training import clip
        if 'vit_B16' in opts.feature_network:
            detector = clip.load('ViT-B/16', device='cpu', jit=False, download_root='/mnt/petrelfs/zhangyichi')[0].visual.eval().requires_grad_(False).to(device)
            _feature_detector_cache[key] = detector
        elif 'vit_B32' in opts.feature_network:
            detector = clip.load('ViT-B/32', device='cpu', jit=False, download_root='/mnt/petrelfs/zhangyichi')[0].visual.eval().requires_grad_(False).to(device)
            _feature_detector_cache[key] = detector
        elif 'vit_L14' in opts.feature_network:
            detector = clip.load('ViT-L/14', device='cpu', jit=False, download_root='/mnt/petrelfs/zhangyichi')[0].visual.eval().requires_grad_(False).to(device)
            _feature_detector_cache[key] = detector
        else:
            detector = clip.load('RN50', device='cpu', jit=False)[0].visual.eval().requires_grad_(False).to(device)
            _feature_detector_cache[key] = detector
    
    elif 'moco' in opts.feature_network and 'moco_v2' not in opts.feature_network:
        if 'vit' in opts.feature_network:
            detector = vits.__dict__['vit_base']()
            linear_keyword='head'
        else:
            detector = models.__dict__['resnet50']()
            linear_keyword='fc'

        checkpoint = torch.load(url, map_location="cpu")
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                # remove prefix
                state_dict[k[len("module.base_encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        msg = detector.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}
        torch.cuda.set_device(device)
        detector=detector.to(device)
        _feature_detector_cache[key]=detector

    elif 'moco_v2' in opts.feature_network:
        detector = models.__dict__['resnet50']()
        checkpoint = torch.load(url, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        msg = detector.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        print("=> loaded pre-trained model '{}'".format(url))
        torch.cuda.set_device(device)
        detector=detector.to(device)
        _feature_detector_cache[key]=detector

    elif 'convnext' in opts.feature_network:
        model_name = "convnext_base"
        detector = create_model(model_name, pretrained=True).to(device)
        _feature_detector_cache[key]=detector
    elif 'vitcls_base' in opts.feature_network:
        model_name = "vit_base_patch16_224"
        detector = create_model(model_name, pretrained=True).to(device)
        _feature_detector_cache[key]=detector
    elif 'swin' in opts.feature_network:
        model_name = "swin_base_patch4_window7_224"
        detector = create_model(model_name, pretrained=True).to(device)
        _feature_detector_cache[key]=detector
    elif 'deit' in opts.feature_network:
        model_name = "deit_base_patch16_224"
        detector = create_model(model_name, pretrained=True).to(device)
        _feature_detector_cache[key]=detector    
    elif 'repvgg' in opts.feature_network:
        model_name = "repvgg_b3"
        detector = create_model(model_name, pretrained=True).to(device)
        _feature_detector_cache[key]=detector  
    elif 'resmlp' in opts.feature_network:
        model_name = "resmlp_24_224_dino"
        detector = create_model(model_name, pretrained=True).to(device)
        _feature_detector_cache[key]=detector  
    else:
        try:
            with dnnlib.util.open_url(url, verbose=(verbose and is_leader)) as f:
                _feature_detector_cache[key] = pickle.load(f).to(device)
        except:
            fn = os.path.split(url)[-1]
            url = f"{cache_path}/{fn}"
            with dnnlib.util.open_url(url, verbose=(verbose and is_leader)) as f: 
                _feature_detector_cache[key] = pickle.load(f).to(device)
    if is_leader and num_gpus > 1:
        torch.distributed.barrier() # others follow
    return _feature_detector_cache[key]

#----------------------------------------------------------------------------

def iterate_random_labels(opts, batch_size):
    if opts.G.c_dim == 0:
        c = torch.zeros([batch_size, opts.G.c_dim], device=opts.device)
        while True:
            yield c
    else:
        dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
        while True:
            c = [dataset.get_label(np.random.randint(len(dataset))) for _i in range(batch_size)]
            c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)
            yield c


#----------------------------------------------------------------------------

class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x, num_gpus=1, rank=0):
        assert isinstance(x, torch.Tensor)  and x.ndim == 2
        assert 0 <= rank < num_gpus
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1) # interleave samples
        self.append(x.cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = FeatureStats(capture_all=s.capture_all, max_items=s.max_items)
        obj.__dict__.update(s)
        return obj

#----------------------------------------------------------------------------

class ProgressMonitor:
    def __init__(self, tag=None, num_items=None, flush_interval=1000, verbose=False, progress_fn=None, pfn_lo=0, pfn_hi=1000, pfn_total=1000):
        self.tag = tag
        self.num_items = num_items
        self.verbose = verbose
        self.flush_interval = flush_interval
        self.progress_fn = progress_fn
        self.pfn_lo = pfn_lo
        self.pfn_hi = pfn_hi
        self.pfn_total = pfn_total
        self.start_time = time.time()
        self.batch_time = self.start_time
        self.batch_items = 0
        if self.progress_fn is not None:
            self.progress_fn(self.pfn_lo, self.pfn_total)

    def update(self,layer, cur_items):
        assert (self.num_items is None) or (cur_items <= self.num_items)
        if (cur_items < self.batch_items + self.flush_interval) and (self.num_items is None or cur_items < self.num_items):
            return
        cur_time = time.time()
        total_time = cur_time - self.start_time
        time_per_item = (cur_time - self.batch_time) / max(cur_items - self.batch_items, 1)
        if (self.verbose) and (self.tag is not None):
            print(f'{self.tag:<19s} {layer:<18s}    items {cur_items:<7d} time {dnnlib.util.format_time(total_time):<12s} ms/item {time_per_item*1e3:.2f}')
        self.batch_time = cur_time
        self.batch_items = cur_items

        if (self.progress_fn is not None) and (self.num_items is not None):
            self.progress_fn(self.pfn_lo + (self.pfn_hi - self.pfn_lo) * (cur_items / self.num_items), self.pfn_total)

    def sub(self, tag=None, num_items=None, flush_interval=1000, rel_lo=0, rel_hi=1):
        return ProgressMonitor(
            tag             = tag,
            num_items       = num_items,
            flush_interval  = flush_interval,
            verbose         = self.verbose,
            progress_fn     = self.progress_fn,
            pfn_lo          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_lo,
            pfn_hi          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_hi,
            pfn_total       = self.pfn_total,
        )

#----------------------------------------------------------------------------
# fwd hook
activation = {}
def getActivation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

#----------------------------------------------------------------------------

def find_real_layer_in_activation(layer):
    name="real_"+layer
    if name in activation:
        return True
    else:
        return False

def find_gen_layer_in_activation(layer):
    name="gen_"+layer
    if name in activation:
        return True
    else:
        return False

def find_file_exists(opt,file):
    flag = os.path.isfile(file) if opt.rank == 0 else False
    if opt.num_gpus > 1:
        flag = torch.as_tensor(flag, dtype=torch.float32, device=opt.device)
        torch.distributed.broadcast(tensor=flag, src=0)
        flag = (float(flag.cpu()) != 0)
    return flag

def hasattr_plus(detector,layer):
    model=detector
    if hasattr(model,layer):
        return True
    else:
        import re
        layer=layer.replace('[','.')
        layer=layer.replace(']','')
        layers=re.split('\.',layer)
        for l in layers:
            if hasattr(model,l):
                model=getattr(model,l)
            else:
                return False
        return True       

def getattr_plus(detector,layer):
    model=detector
    if hasattr(model,layer):
        return getattr(model,layer)
    else:
        import re
        layer=layer.replace('[','.')
        layer=layer.replace(']','')
        layers=re.split('\.',layer)
        for l in layers:
            if hasattr(model,l):
                model=getattr(model,l)
            else:
                raise AttributeError
        return model

def seed_everything(seed):
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)

def random_projection(features):
    insize=features.shape[1]
    outsize=int(insize/2)
    fc_name=str(insize)
    seed_everything(0)
    fc=torch.nn.Linear(insize,outsize).to('cuda')
    torch.nn.init.normal_(fc.weight, mean=0.0, std=0.01)
    return_features=fc(features)
    return return_features
    

def pre_process(feature_network):
    if feature_network is None:
        preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    elif feature_network == 'spr' or feature_network == 'swav':
        preprocess = transforms.Compose([
        #transforms.Resize(256),
        #transforms.CenterCrop(224),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    elif 'clip' in feature_network:
        preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    elif feature_network == 'fasterrcnn_r50' or feature_network == 'maskrcnn_r50':
        preprocess = transforms.Compose([
        #transforms.Resize(224),
        transforms.ToTensor(),
    ]) 
    elif 'moco' in feature_network or 'vgg' in feature_network:
        preprocess =transforms.Compose([
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif 'convnext' in feature_network:
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
    elif 'vitcls_base' in feature_network or 'deit' in feature_network:
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
    elif 'swin' in feature_network or 'resmlp' in feature_network:
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
    elif 'repvgg' in feature_network:
        from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        NORMALIZE_MEAN = IMAGENET_DEFAULT_MEAN
        NORMALIZE_STD = IMAGENET_DEFAULT_STD
        preprocess =transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]) 
    else:
        preprocess=None
    return preprocess


def image_preprocess(images,opts,preprocess):
    if preprocess is None:
        #if opts.feature_network == 'clip':
            #images = images.to(opts.device).to(torch.float32) / 127.5 - 1
        #else:
        images=images.to(torch.float32)
    else:
        new_images=None
        for image in images:
            new_img = transforms.ToPILImage()(image).convert('RGB')
            image=preprocess(new_img).unsqueeze(0).to(opts.device)
            if new_images is None:
                new_images=image
            else:
                new_images=torch.cat((new_images,image),0).to(opts.device)
        images=new_images.to(opts.device)
    return images

def post_process(feature,opts):
    assert opts.post_process in ['mean', 'max', 'min', 'var', 'none'], f'post_process must be one of (mean, max, min, var,none)!'
    # mean
    if opts.post_process == 'mean':
        if opts.feature_network is None:
            output=feature.mean([2,3])
        elif 'vit' in opts.feature_network:
            if 'clip' in opts.feature_network:
                feature=feature.permute(1,0,2)
            if opts.token_vit:
                feature_cut=feature[:,:1,:]
            else:
                #feature_cut=feature        
                feature_cut=feature[:,1:,:]
            output=feature_cut.mean(dim=1)
        elif 'swin' in opts.feature_network or 'resmlp' in opts.feature_network or 'deit' in opts.feature_network:
            output = feature.mean(dim=1)
        else:
            output=feature.mean([2,3])
    # max
    elif opts.post_process =='max':
        if opts.feature_network is None:
            output = torch.max(feature, dim=2)[1]
            output = torch.max(output, dim=2)[1]
        elif 'vit' in opts.feature_network:
            if 'clip' in opts.feature_network:
                feature=feature.permute(1,0,2)
            if opts.token_vit:
                feature_cut=feature[:,:1,:]
                output=feature_cut.mean(dim=1)
            else:        
                feature_cut=feature[:,1:,:]
                output=torch.max(feature_cut, dim=2)[1]
        elif 'swin' in opts.feature_network or 'resmlp' in opts.feature_network or 'deit' in opts.feature_network:
            output = feature.mean(dim=1)
        else:
            output = torch.max(feature, dim=2)[1]
            output = torch.max(output, dim=2)[1]
    # min
    elif opts.post_process =='min':
        if opts.feature_network is None:
            output = torch.min(feature, dim=2)[1]
            output = torch.min(output, dim=2)[1]
        elif 'vit' in opts.feature_network:
            if 'clip' in opts.feature_network:
                feature=feature.permute(1,0,2)
            if opts.token_vit:
                feature_cut=feature[:,:1,:]
                output=feature_cut.mean(dim=1)
            else:        
                feature_cut=feature[:,1:,:]
                output=torch.min(feature_cut, dim=2)[1]
        elif 'swin' in opts.feature_network or 'resmlp' in opts.feature_network  or 'deit' in opts.feature_network:
            output = feature.mean(dim=1)
        else:
            output = torch.min(feature, dim=2)[1]
            output = torch.min(output, dim=2)[1]
    # variance
    elif opts.post_process =='var':
        if opts.feature_network is None:
            output=torch.var(feature, dim=[2,3])
        elif 'vit' in opts.feature_network:
            if 'clip' in opts.feature_network:
                feature=feature.permute(1,0,2)
            if opts.token_vit:
                feature_cut=feature[:,:1,:]
                output=feature_cut.mean(dim=1)
            else:        
                feature_cut=feature[:,1:,:]
                output=torch.var(feature_cut,dim=2)
        elif 'swin' in opts.feature_network or 'resmlp' in opts.feature_network  or 'deit' in opts.feature_network:
            output = feature.mean(dim=1)
        else:
            output=torch.var(feature, dim=[2,3])
    else:
        if opts.feature_network is None:
            output=feature
        elif 'vit' in opts.feature_network:
            if 'clip' in opts.feature_network:
                feature=feature.permute(1,0,2)
            output=feature[:,1:,:]
        else:
            output=feature
    return output

#----------------------------------------------------------------------------

# def softmax3d(input, feature_network):
#     m = nn.Softmax()
#     if feature_network is not None and 'vit' in feature_network:
#         a, b, c = input.size()
#         input = torch.reshape(input, (1, -1))
#         output = m(input)
#         output = torch.reshape(output, (a, b, c))
#     else:
#         a, b, c, d = input.size()
#         input = torch.reshape(input, (1, -1))
#         output = m(input)
#         output = torch.reshape(output, (a, b, c, d))
#     return output

def compute_feature_stats_for_dataset(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1,batch_size=1, data_loader_kwargs=None, max_items=None,layers=None, **stats_kwargs):
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)
    stats={}
    num_items = len(dataset)
    #dnnlib.set_cache_dir(cache_path)
    if max_items is not None:
        num_items = min(num_items, max_items)
    print("dataset_num_items=%d" %num_items)
    #preprocess
    preprocess=pre_process(opts.feature_network)

    #get_detector
    progress = opts.progress.sub(tag='dataset features', num_items=num_items, rel_lo=rel_lo, rel_hi=rel_hi)
    if opts.feature_network == 'spr':
        detector=torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).to(opts.device)
    elif opts.feature_network == 'vgg':
        detector=torchvision.models.vgg19(pretrained=True).to(opts.device)
    elif opts.feature_network == 'fasterrcnn_r50':
        detector=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(opts.device)
    elif opts.feature_network == 'maskrcnn_r50':
        detector=torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(opts.device)
    elif opts.feature_network is None or opts.feature_network == 'inception':
        detector=torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True).to(opts.device)
    else:
        detector = get_feature_detector(url=detector_url, opts=opts, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    detector.eval()
    layer_items=len(layers)
    #print(detector)
    # Try to lookup from feature_save.
    feature_files={}
    for layer in layers:
    # Choose cache file name.
        args = dict(dataset_kwargs=opts.dataset_kwargs, detector_url=detector_url, detector_kwargs=detector_kwargs, stats_kwargs=stats_kwargs)
        if opts.feature_network is None:
            feature_network = 'inception'
        else:
            feature_network = opts.feature_network
        if opts.token_vit == True:
            feature_tag = f'{dataset.name}-{feature_network}-{layer}-token'
        else:
            feature_tag = f'{dataset.name}-{feature_network}-{layer}'
        feature_real = os.path.join(opts.feature_save,'real')
        feature_files[layer] = os.path.join(feature_real, feature_tag + '.pkl')
        # Check if the file exists (all processes must agree).
        flag = find_file_exists(opts,feature_files[layer])
        # Load.
        if flag and opts.cache:
            stats[layer]=FeatureStats.load(feature_files[layer])
            layer_items-=1
    hook={}
    # Activation_Initialize.
    if layer_items!=0:
        for layer in layers:
            if layer not in stats:
                assert hasattr_plus(detector,layer), 'Layer Error!'
                hook_name="real_"+layer
                hook[layer]=getattr_plus(detector,layer).register_forward_hook(getActivation(hook_name))
                stats[layer]=FeatureStats(max_items=num_items, **stats_kwargs)
        item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]        
        features={}
        feature_save={}
        # Main loop.
        for images, _labels in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs):

            if images.shape[1] == 1:
                images = images.repeat([1, 3, 1, 1])
        
            with torch.no_grad():
                images=image_preprocess(images,opts,preprocess)#preprocess
                feature_fwd = detector(images.float().to(opts.device))#forward
                for layer in layers:
                    if find_real_layer_in_activation(layer):
                        hook_name="real_"+layer
                        # post process
                        if opts.post_process is not None:
                            features[layer] = post_process(activation[hook_name],opts)
                for layer in layers:
                    if find_real_layer_in_activation(layer):
                        stats[layer].append_torch(features[layer], num_gpus=opts.num_gpus, rank=opts.rank)
                        progress.update(layer,stats[layer].num_items)
                
        if opts.feature_save_flag == True:
            for layer in layers:
                if find_real_layer_in_activation(layer):
                    # Save to cache.
                    if feature_files[layer] is not None:
                        os.makedirs(os.path.dirname(feature_files[layer]), exist_ok=True)
                        temp_file = feature_files[layer] + '.' + uuid.uuid4().hex
                        stats[layer].save(temp_file)
                        os.replace(temp_file, feature_files[layer]) # atomic    
    for h in hook.keys():  
        hook[h].remove()
    return stats 

def compute_feature_stats_for_generate_dataset(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1,batch_size=1, generate_data_loader_kwargs=None, max_items=None,layers=None, **stats_kwargs):
    generate_dataset = dnnlib.util.construct_class_by_name(**opts.generate_dataset_kwargs)
    if generate_data_loader_kwargs is None:
        generate_data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)
    stats={}
    # dnnlib.set_cache_dir(cache_path)
    # num_items = len(generate_dataset)
    if max_items is not None:
        num_items = min(len(generate_dataset), max_items)
    else:
        num_items = len(generate_dataset)
    # if max_items is not None:
    #     num_items = min(num_items, max_items)
    print("generate_num_items=%d" %num_items)
  
    #preprocess
    preprocess=pre_process(opts.feature_network)
    #get_detector
    progress = opts.progress.sub(tag='generate_dataset features', num_items=num_items, rel_lo=rel_lo, rel_hi=rel_hi)
    if opts.feature_network == 'spr':
        detector=torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).to(opts.device)
    elif opts.feature_network == 'vgg':
        detector=torchvision.models.vgg19(pretrained=True).to(opts.device)
    elif opts.feature_network == 'fasterrcnn_r50':
        detector=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(opts.device)
    elif opts.feature_network == 'maskrcnn_r50':
        detector=torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(opts.device)
    elif opts.feature_network is None or opts.feature_network == 'inception':
        detector=torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True).to(opts.device)
    else:
        detector = get_feature_detector(url=detector_url, opts=opts, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    detector.eval()
    layer_items=len(layers)

    # Try to lookup from feature_save.
    feature_files={}
    for layer in layers:
    # Choose cache file name.
        args = dict(generate_dataset_kwargs=opts.generate_dataset_kwargs, detector_url=detector_url, detector_kwargs=detector_kwargs, stats_kwargs=stats_kwargs)
        if opts.feature_network is None:
            feature_network = 'inception'
        else:
            feature_network = opts.feature_network
        if opts.token_vit == True:
            feature_tag = f'{generate_dataset.name}-{feature_network}-{layer}-token'
        else:
            feature_tag = f'{generate_dataset.name}-{feature_network}-{layer}'
        feature_gen = os.path.join(opts.feature_save,'generate')
        feature_files[layer] = os.path.join(feature_gen, feature_tag + '.pkl')
        # Check if the file exists (all processes must agree).
        flag = find_file_exists(opts,feature_files[layer])
        # Load.
        if flag and opts.cache:
            stats[layer]=FeatureStats.load(feature_files[layer])
            layer_items-=1
    hook={}
    # Activation_Initialize.
    if layer_items!=0:
        for layer in layers:
            assert hasattr_plus(detector,layer), 'Layer Error!'
            hook_name="gen_"+layer
            hook[layer]=getattr_plus(detector,layer).register_forward_hook(getActivation(hook_name))
            stats[layer]=FeatureStats(max_items=num_items, **stats_kwargs)
        item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]        
        if opts.random == True:
            sampler=torch.utils.data.RandomSampler(generate_dataset, replacement=True, num_samples=opts.random_size)
        else:
            sampler=item_subset
        features={}
        feature_save={}
        # Main loop.
        for images, _labels in torch.utils.data.DataLoader(dataset=generate_dataset, sampler=sampler, batch_size=batch_size, **generate_data_loader_kwargs):
            if images.shape[1] == 1:
                images = images.repeat([1, 3, 1, 1])
        
            with torch.no_grad():
                images=image_preprocess(images,opts,preprocess)#preprocess
                # detector = detector.cuda()
                feature_fwd = detector(images.to(opts.device))#forward 

            for layer in layers:
                if find_gen_layer_in_activation(layer):
                    hook_name="gen_"+layer
                    # post process
                    if opts.post_process is not None:
                        features[layer] = post_process(activation[hook_name],opts)
            for layer in layers:
                if find_gen_layer_in_activation(layer):
                    stats[layer].append_torch(features[layer], num_gpus=opts.num_gpus, rank=opts.rank)
                    progress.update(layer,stats[layer].num_items)
        if opts.feature_save_flag == True:
            for layer in layers:
                if find_gen_layer_in_activation(layer):
                    # Save to cache.
                    if feature_files[layer] is not None:
                        os.makedirs(os.path.dirname(feature_files[layer]), exist_ok=True)
                        temp_file = feature_files[layer] + '.' + uuid.uuid4().hex
                        stats[layer].save(temp_file)
                        os.replace(temp_file, feature_files[layer]) # atomic
    for h in hook.keys():  
        hook[h].remove()
    return stats


def compute_feature_stats_for_generator(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=1, batch_gen=None, max_items=None, layers=None, **stats_kwargs):
    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0
    # Setup generator and labels.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    c_iter = iterate_random_labels(opts=opts, batch_size=batch_gen)
    #dnnlib.set_cache_dir(cache_path)
    # Initialize.
    stats={}
        
    assert max_items is not None
    progress = opts.progress.sub(tag='generator features', num_items=max_items, rel_lo=rel_lo, rel_hi=rel_hi)
    preprocess=pre_process(opts.feature_network)
    if opts.feature_network == 'spr':
        detector=torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).to(opts.device)
    elif opts.feature_network == 'vgg':
        detector=torchvision.models.vgg19(pretrained=True).to(opts.device)
    elif opts.feature_network == 'fasterrcnn_r50':
        detector=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(opts.device)
    elif opts.feature_network == 'maskrcnn_r50':
        detector=torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(opts.device)
    elif opts.feature_network is None or opts.feature_network == 'inception':
        detector=torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True).to(opts.device)
    else:
        detector = get_feature_detector(url=detector_url, opts=opts, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)
    detector.eval()
    hook={}
    # Main loop.
    for layer in layers:
        stats[layer] = FeatureStats(max_items=max_items,**stats_kwargs)
        stat=stats[layer]
        hook_name="gen_"+layer
        hook[layer]=getattr_plus(detector,layer).register_forward_hook(getActivation(hook_name))
    features={}
    feature_save={}
    while not stat.is_full():
        images = []
        for _i in range(batch_size // batch_gen):
            #c_iter = iterate_random_labels(opts=opts, batch_size=batch_gen)
            z = torch.randn([batch_gen, G.z_dim], device=opts.device)
            img = G(z=z, c=next(c_iter), **opts.G_kwargs)
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            images.append(img)
        images = torch.cat(images)
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])

        with torch.no_grad():
            images=image_preprocess(images,opts,preprocess)#preprocess
            feature_fwd = detector(images.to(opts.device))#forward
            for layer in layers:
                if find_gen_layer_in_activation(layer):
                    hook_name="gen_"+layer
                    # post process
                    if opts.post_process is not None:
                        features[layer] = post_process(activation[hook_name],opts)
        for layer in layers:
                if find_gen_layer_in_activation(layer):
                    stats[layer].append_torch(features[layer], num_gpus=opts.num_gpus, rank=opts.rank)
                    progress.update(layer, stats[layer].num_items)
    for h in hook.keys():  
        hook[h].remove()
    return stats
#---------------------------------------------------------------------------- 
