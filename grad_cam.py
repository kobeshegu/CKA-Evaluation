from inspect import getattr_static
import os
from pyexpat import features
import time
import re
import hashlib
import pickle
import copy
import click
from tkinter import W
from einops import rearrange
import uuid
import numpy as np
import torch
import math
import dnnlib
import torchvision
import scipy.linalg
import random
import cv2
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from torchsummary import summary
from sqrtm import sqrtm
from visualizer import *
from mocov3 import vits
import torchvision.models as torch_models
from sklearn.random_projection import GaussianRandomProjection
from calc_metrics import parse_comma_separated_list
import torch.nn as nn
from timm import create_model

detector_urls=dict(
    swav = './detector/swav.pth.tar',
    clip = './detector/clip.pt',
    spr = './detector/spr.pth.tar',
    clip_vit_B16 = './detector/clip_vit_B16.pt',
    clip_vit_B32 = './detector/clip_vit_B32.pt',
    moco_v2_r50_i = './detector/moco_v2_r50_imagenet.pth.tar',
    moco_r50_i = './detector/moco_r50_imagenet.pth.tar',
    moco_r50_f = './detector/moco_r50_ffhq.pth.tar',
    moco_r50_anime = './detector/moco_r50_anime.pth.tar',
    moco_r50_celeba = './detector/moco_r50_celeba.pth.tar',
    moco_r50_church = './detector/moco_r50_church.pth.tar',
    moco_r50_afhq = './detector/moco_r50_afhq.pth.tar',
    moco_r50_lsunhorse = './detector/moco_r50_lsunhorse.pth.tar',
    moco_r50_lsunbedroom = './detector/moco_r50_lsunbedroom.pth.tar',
    moco_r50_s0 = './detector/moco_r50_s0.pth.tar',
    moco_r50_s1 = './detector/moco_r50_s1.pth.tar',
    moco_r50_s2 = './detector/moco_r50_s2.pth.tar',
    moco_vit_i = './detector/moco_vit_imagenet.pth.tar',
    moco_vit_f = './detector/moco_vit_ffhq.pth.tar',
    moco_vit_anime = './detector/moco_vit_anime.pth.tar',
    moco_vit_celeba = './detector/moco_vit_celeba.pth.tar',
    moco_vit_church = './detector/moco_vit_church.pth.tar',
    moco_vit_afhq = './detector/moco_vit_afhq.pth.tar',
    moco_vit_lsunhorse = './detector/moco_vit_lsunhorse.pth.tar',
    moco_vit_lsunbedroom = './detector/moco_vit_lsunbedroom.pth.tar',
    moco_vit_s0 = './detector/moco_vit_s0.pth.tar',
    moco_vit_s1 = './detector/moco_vit_s1.pth.tar',
    moco_vit_s2 = './detector/moco_vit_s2.pth.tar',
    convnext_base = './detector/convnext_base_1k_224_ema.pth'
    #vitcls_base_patch16_224 = './detector/'
)

models=["inception","resnet50","clip","clip_vit_B16","clip_vit_B32","clip_vit_L14","swav","fasterrcnn_r50",
        "maskrcnn_r50","vgg19","moco_v2_r50_i","moco_r50_i","moco_r50_f","moco_r50_s0","moco_r50_anime","moco_r50_celeba",
        "moco_r50_church","moco_r50_afhq","moco_r50_lsunhorse",
        "moco_r50_lsunbedroom","moco_vit_i","moco_vit_f","moco_vit_s0","moco_vit_anime","moco_vit_celeba","moco_vit_church",
        "moco_vit_afhq","moco_vit_lsunhorse","moco_vit_lsunbedroom","convnext_base","vitcls_base_patch16_224",
        "swintransformer", "repvgg", "resmlp", "deit", "mixer_b16_224", "gmlp_s16_224", "maxxvitv2_rmlp_base_rw_224"]
#layer1=["layer1","layer2","layer3","layer4"]
layer1=["layer1.0","layer1.1","layer1.2","layer2.0","layer2.1","layer2.2","layer2.3","layer3.0","layer3.1","layer3.2","layer3.3","layer3.4","layer3.5","layer4.0","layer4.1","layer4.2"]
#layer2=["Conv2d_4a_3x3","Mixed_5d","Mixed_6e","Mixed_7c"]
layer2=["Conv2d_1a_3x3","Conv2d_2a_3x3","Conv2d_2b_3x3","Conv2d_3b_1x1","Conv2d_4a_3x3","Mixed_5b","Mixed_5c","Mixed_5d","Mixed_6a","Mixed_6b","Mixed_6c","Mixed_6d","Mixed_6e","Mixed_7a","Mixed_7b","Mixed_7c"]
layer3=["blocks.0","blocks.1","blocks.2","blocks.3","blocks.4","blocks.5","blocks.6","blocks.7","blocks.8","blocks.9","blocks.10","blocks.11",]
layer4=["backbone.body.layer1","backbone.body.layer2","backbone.body.layer3","backbone.body.layer4"]
layer5=["features.9","features.18","features.27","features.36"]
layer6=["transformer.resblocks.0","transformer.resblocks.1","transformer.resblocks.2","transformer.resblocks.3","transformer.resblocks.4","transformer.resblocks.5","transformer.resblocks.6","transformer.resblocks.7","transformer.resblocks.8","transformer.resblocks.9","transformer.resblocks.10","transformer.resblocks.11"]
layer7=["transformer.resblocks.1","transformer.resblocks.3","transformer.resblocks.5","transformer.resblocks.7","transformer.resblocks.9","transformer.resblocks.11","transformer.resblocks.13","transformer.resblocks.15","transformer.resblocks.17","transformer.resblocks.19","transformer.resblocks.21","transformer.resblocks.23"]
layer8=["stages.0","stages.1","stages.2","stages.3"]
layer9 = ["blocks.0","blocks.1","blocks.2","blocks.3","blocks.4","blocks.5","blocks.6","blocks.7","blocks.8","blocks.9","blocks.10","blocks.11","blocks.12","blocks.13","blocks.14","blocks.15","blocks.16","blocks.17","blocks.18","blocks.19","blocks.20","blocks.21","blocks.22","blocks.23"]
layer10=["layers.0","layers.1","layers.2","layers.3"]
layer11 = ["stages.0","stages.1","stages.2","stages.3"]
layergmlp = ["blocks.0","blocks.1","blocks.2","blocks.3","blocks.4","blocks.5","blocks.6","blocks.7","blocks.8","blocks.9","blocks.10","blocks.11","blocks.12","blocks.13","blocks.14","blocks.15","blocks.16","blocks.17","blocks.18","blocks.19","blocks.20","blocks.21","blocks.22","blocks.23","blocks.24","blocks.25","blocks.26","blocks.27","blocks.28","blocks.29"]
cache_path = '~/.cache'
_feature_detector_cache = dict()

def get_feature_detector_name(url):
    return os.path.splitext(url.split('/')[-1])[0]

def get_feature_detector(url, feature_network, device=torch.device('cpu'), num_gpus=1, rank=0, verbose=False):
    assert 0 <= rank < num_gpus
    key = (url, device)
    is_leader = (rank == 0)
    if not is_leader and num_gpus > 1:
        torch.distributed.barrier() # leader goes first
    if url is not None:
        fn = os.path.split(url)[-1]
    if 'swav' in feature_network:
        from training.resnet_swav import resnet50
        detector = resnet50(url=url).eval().requires_grad_(False).to(device)
        _feature_detector_cache[key] = detector
    elif 'clip' in feature_network:
        from training import clip
        if 'vit_B16' in feature_network:
            detector = clip.load('ViT-B/16', device='cpu', jit=False, download_root='/mnt/petrelfs/zhangyichi')[0].visual.eval().requires_grad_(False).to(device)
            _feature_detector_cache[key] = detector
        elif 'vit_B32' in feature_network:
            detector = clip.load('ViT-B/32', device='cpu', jit=False, download_root='/mnt/petrelfs/zhangyichi')[0].visual.eval().requires_grad_(False).to(device)
            _feature_detector_cache[key] = detector
        else:
            detector = clip.load('RN50', device='cpu', jit=False)[0].visual.eval().requires_grad_(False).to(device)
            _feature_detector_cache[key] = detector
    elif 'moco' in feature_network and 'moco_v2' not in feature_network:
        if 'vit' in feature_network:
            detector = vits.__dict__['vit_base']()
            linear_keyword='head'
        else:
            detector = torch_models.__dict__['resnet50']()
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
    elif 'moco_v2' in feature_network:
        detector = torch_models.__dict__['resnet50']()
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
    elif 'convnext' in feature_network:
        model_name = "convnext_base"
        detector = create_model(model_name, pretrained=True).to(device)
        _feature_detector_cache[key]=detector
    elif 'vitcls' in feature_network:
        model_name = "vit_base_patch16_224"
        detector = create_model(model_name, pretrained=True).to(device)
        _feature_detector_cache[key]=detector
    elif 'swin' in feature_network:
        model_name = "swin_base_patch4_window7_224"
        detector = create_model(model_name, pretrained=True).to(device)
        _feature_detector_cache[key]=detector
    elif 'deit' in feature_network:
        model_name = "deit_base_patch16_224"
        detector = create_model(model_name, pretrained=True).to(device)
        _feature_detector_cache[key]=detector 
    elif 'repvgg' in feature_network:
        model_name = "repvgg_b3"
        detector = create_model(model_name, pretrained=True).to(device)
        _feature_detector_cache[key]=detector  
    elif 'resmlp' in feature_network:
        model_name = "resmlp_24_224_dino"
        detector = create_model(model_name, pretrained=True).to(device)
        _feature_detector_cache[key]=detector
    elif 'mixer' in feature_network:
        model_name = "mixer_b16_224"
        detector = create_model(model_name, pretrained=True).to(device)
        _feature_detector_cache[key] = detector
    elif 'gmlp' in feature_network:
        model_name = "gmlp_s16_224"
        detector = create_model(model_name, pretrained=True).to(device)
        _feature_detector_cache[key] = detector
    elif 'rmlp' in feature_network:
        model_name = "maxxvitv2_rmlp_base_rw_224"
        detector = create_model(model_name, pretrained=True).to(device)
        _feature_detector_cache[key] = detector
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

grad_block={}
#bwd hook
def getGrad(name):
    def back_hook(grad):
        #print(grad)
        grad_block[name] = grad.detach().requires_grad_(True)
    return back_hook

# fwd hook
activation = {}
def getActivation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def show_cam_on_image(img, mask,savename, if_show=False, if_write=False):
    heatmap = mask
    img=img[:,:,::-1]
    cam = heatmap + np.float64(img)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    whole_cam = cam[:,:,::-1]
    if if_write:
        cv2.imwrite(savename, cam)
    if if_show:
        plt.imshow(cam[:,:,::-1])
        plt.show()
    return whole_cam

#load data
rank=0
device = 'cuda'

def pre_process(feature_network):
    if feature_network =='resnet50' or feature_network =='swav':
        preprocess = transforms.Compose([
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
        #weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        #preprocess = weights.transforms()
        preprocess = transforms.Compose([
        #transforms.Resize(224),
        transforms.ToTensor(),
    ]) 
    elif feature_network =='inception':
        preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    elif 'moco' in feature_network or 'vgg19' in feature_network:
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
    elif 'swin' in feature_network or 'resmlp' in feature_network or 'mixer' in feature_network or 'gmlp' in feature_network or 'rmlp' in feature_network:
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

def softmax3d(input, softmax_mode):
    assert softmax_mode in ['3d', '2d', 'L1', 'L2'], f'cka_normalize dimension must be one of `3d`, `2d`, `L1` and `L2` !'
    # feature normalization
    m = nn.Softmax()
    if softmax_mode == '3d':
        N, C = input.size()
        input = torch.reshape(input, (1, -1))
        output = m(input)
        output = torch.reshape(output, (N, C))
    elif softmax_mode == '2d':
        output = m(input)
    elif softmax_mode == 'L1':
        output = torch.nn.functional.normalize(input, p=1.0, dim=1)
    else:
        output = torch.nn.functional.normalize(input, p=2.0, dim=1)
    return output

def fid_cal(N,mu_real,mu_gen,sigma_real,sigma_gen,feature):
    mu_new_gen=(mu_gen*(N-1)/N+feature/N)
    fmu=(feature-mu_gen)
    sigma_new_gen=(sigma_gen*(N-2)/(N-1)+torch.matmul(fmu.T,fmu)/N)

    m = torch.square(mu_real-mu_new_gen).sum()
    s = sqrtm(torch.matmul(sigma_real,sigma_new_gen))
    fid_res = (m + torch.trace(sigma_real + sigma_new_gen - s*2))
    return fid_res

def kid_cal(real_features=None, gen_features=None, feature=None, max_subset_size=1000, num_subsets=100):
    n = real_features.shape[1]
    m = min(min(real_features.shape[0], gen_features.shape[0]+1), max_subset_size)
    t = 0

    pre_x = gen_features[np.random.choice(gen_features.shape[0], m-1, replace=False)]
    y = real_features[np.random.choice(real_features.shape[0], m, replace=False)]
    x =torch.cat((pre_x,feature), axis=0)
    a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
    b = (x @ y.T / n + 1) ** 3
    t += (a.sum() - torch.diag(a).sum()) / (m - 1) - b.sum() * 2 / m

    gen_feature_all=torch.cat((gen_features,feature), axis=0)
    for _subset_idx in range(num_subsets-1):
        x = gen_feature_all[np.random.choice(gen_feature_all.shape[0], m, replace=False)]
        y = real_features[np.random.choice(real_features.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - torch.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    kid_res = t / num_subsets / m
    torch.cuda.empty_cache()
    return kid_res

def centering(K):
    n = K.shape[0]
    unit = torch.ones([n, n], device=K.device)
    I = torch.eye(n, device=K.device)
    H = I - unit / n
    #return torch.matmul(torch.matmul(H, K), H)
    return torch.matmul(K, H)

def rbf_kernel(GX):
   # GX = torch.matmul(GX, GX.T)
    KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
    mdist = torch.median(KX[KX!=0])
    sigma = math.sqrt(mdist)
    KX = KX * (-0.5 / (sigma * sigma))
    KX = torch.exp(KX)
    return KX

def poly_kernel(GX,  m, poly_constant=0, poly_power=3):
    return ( (poly_constant + torch.matmul(GX, GX.T) / m ) ** poly_power ) / ( 10 **6 ) # divide m**6 to avoid Inf issues

def cka_cal(real_features=None, gen_features=None, feature=None, max_subset_size=5000, num_subsets=50, cka_normalize='2d', dimension='C', kernel='rbf', fusion_softmax_order='pre_fusion'):
    m = min(min(real_features.shape[0], gen_features.shape[0]), max_subset_size)
    cka = 0
    whole_real_features=real_features
    whole_gen_features=torch.cat((gen_features,feature), axis=0)
    if cka_normalize is not None and fusion_softmax_order == 'post_fusion':
        assert cka_normalize in ['3d', '2d', 'L1', 'L2'], f'cka_normalize dimension must be one of `3d`, `2d`, `L1` and `L2` !'
        whole_real_features = softmax3d(whole_real_features, softmax_mode=cka_normalize)
        whole_gen_features = softmax3d(whole_gen_features, softmax_mode=cka_normalize)

    for _subset_idx in range(num_subsets):
        if _subset_idx == 0:
            p=[]
            for i in range(49999):
                p.append(0.0000001)
            p.append(0.9950001)
            x = whole_gen_features[np.random.choice(whole_gen_features.shape[0], m, replace=False, p=p)]
        else:
            x = whole_gen_features[np.random.choice(whole_gen_features.shape[0], m, replace=False)]
        y = whole_real_features[np.random.choice(whole_real_features.shape[0], m, replace=False)]
        if dimension =='N':
            L_real_features = torch.matmul(y, y.T)
            L_gen_features = torch.matmul(x, x.T)
        else:
            L_real_features = torch.matmul(y.T, y)
            L_gen_features = torch.matmul(x.T, x)
        # kernel type 
        if kernel == 'rbf':
            L_real_features = rbf_kernel(L_real_features)
            L_gen_features = rbf_kernel(L_gen_features)
        elif kernel == 'poly':
            L_real_features = poly_kernel(L_real_features, m=m)
            L_gen_features = poly_kernel(L_gen_features, m=m)
        centering_real_features = centering(L_real_features)  # KH
        centering_gen_features = centering(L_gen_features)  # LH
        hsic = torch.sum(centering_real_features * centering_gen_features)  # trace property: sum of element-wise multiplication = trace(matrix multiplication)
        var1 = torch.sqrt(torch.sum(centering_real_features * centering_real_features))
        var2 = torch.sqrt(torch.sum(centering_gen_features * centering_gen_features))
        cka += hsic / (var1 * var2)
    cka_res = cka / num_subsets
    return cka_res
    
def zero_one_scaling(image: np.ndarray) -> np.ndarray:
    """Scales an image to range [0, 1]."""
    if np.all(image == 0):
        return image
    image = image.astype(np.float32)
    return (image - image.min()) / (image.max() - image.min())


def show_heatmap_on_image(image: np.ndarray,
                              sensitivity_map: np.ndarray,
                              colormap: int = cv2.COLORMAP_JET,#PARULA
                              heatmap_weight: float = 1.5) -> np.ndarray:
    """Overlay the sensitivity map on the image."""
    # Convert sensitivity map to a heatmap.
    heatmap = cv2.applyColorMap(sensitivity_map, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.astype(np.float32) / 255

    # Overlay original RGB image and heatmap with specified weights.
    scaled_image = zero_one_scaling(image=image)
    overlay = heatmap_weight * heatmap.transpose(2, 0, 1) + scaled_image.transpose(2, 0, 1)
    overlay = zero_one_scaling(image=overlay)

    return np.clip(overlay * 255, 0.0, 255.0).astype(np.uint8)

def seed_everything(seed):
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)

def fusion_features(layers=None, features=None, group=0, group_index=None, fusion_ways=None, fusion_softmax_order=None, random_projection=None, cka_normalize='2d', last=None):
    assert fusion_ways in ['sum', 'cat'], f'fusion_ways must be `sum` or `cat`!'
    # group_index = group_index.split(',')
    length = len(features)

    # sum operation
    if fusion_ways == 'sum':
        if group == 0:
            fused_feature = features[layers[0]]
            print(0)
            for index in range(0,int(float(group_index[0]))):
                print(index+1)
                fused_feature += features[layers[index+1]]
        else:
            if last:
                fused_feature = features[layers[int(float(group_index[group-1]))+1]]
                print(int(float(group_index[group-1]))+1)
                for index in range(int(float(group_index[group-1])) + 2, length):
                    print(index)
                    fused_feature += features[layers[index]]
            else:
                fused_feature = features[layers[int(float(group_index[group-1]))+1]]
                print(int(float(group_index[group-1]))+1)
                for index in range(int(float(group_index[group-1])) + 2, int(float(group_index[group])) + 1):
                    print(index)
                    fused_feature += features[layers[index]]

    else:
        if group == 0:
            fused_feature = features[layers[0]]
            print(0)
            for index in range(0,int(float(group_index[0]))):
                print(index+1)
                fused_feature = torch.cat((fused_feature, features[layers[index+1]]), dim=1)
        else:
            if last:
                fused_feature = features[layers[int(float(group_index[group-1]))+1]]
                print(int(float(group_index[group-1]))+1)
                for index in range(int(float(group_index[group-1])) + 2, length):
                    print(index)
                    fused_feature = torch.cat((fused_feature, features[layers[index]]), dim=1)
            else:
                fused_feature = features[layers[int(float(group_index[group-1]))+1]]
                print(int(float(group_index[group-1]))+1)
                for index in range(int(float(group_index[group-1])) + 2, int(float(group_index[group])) + 1):
                    print(index)
                    fused_feature = torch.cat((fused_feature, features[layers[index]]), dim=1)
    return fused_feature

def random_projection(features, out_channels):
    in_channels = features.shape[1]
    out_channels = out_channels
    seed_everything(0)
    fc_layer = torch.nn.Linear(in_channels, out_channels).to('cuda')
    torch.nn.init.normal_(fc_layer.weight, mean=0.0, std=0.01)
    return_features = fc_layer(features.to('cuda'))
    return return_features

def grad_cam(new_generated_image,resolution,feature_network,layers,stats_path,num_gen,metrics,groups,group_index,fusion_ways,fusion_softmax_order,random_projection,fuse_all,fuse_all_ways):
    device = torch.device('cuda', rank)
    image=Image.open(new_generated_image)
    input_batch=None
    detector=None
    
    #load detector
    preprocess = pre_process(feature_network)

    if feature_network=='inception':
        detector=torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    elif feature_network=='resnet50':
        detector=torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    elif feature_network == 'vgg19':
        detector=torchvision.models.vgg19(pretrained=True)
    elif feature_network == 'fasterrcnn_r50':
        detector=torch_models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    elif feature_network == 'maskrcnn_r50':
        detector=torch_models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # elif feature_network == 'vitcls_base_patch16_224':
    #     model_name = "vit_base_patch16_224"
    #     detector = create_model(model_name, pretrained=True)
    else:
        if feature_network not in detector_urls.keys():
            detector_url = None
        else:
            detector_url = detector_urls[feature_network]
        detector = get_feature_detector(url=detector_url, feature_network=feature_network,device=device, num_gpus=1, rank=rank)
    
    detector.eval()
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    if torch.cuda.is_available():
        input_batch=input_batch.to('cuda')
        detector.to('cuda')
    fhooks=[]
    bhooks=[]
    for layer in layers:
        assert hasattr_plus(detector,layer), 'Layer Error!'
        fhook_name=layer
        fhook=getattr_plus(detector,layer).register_forward_hook(getActivation(fhook_name))
        fhooks.append(fhook)
    #get features
    feature_fwd = detector(input_batch)
    # Activations
    feature_map={}
    feature_post={}
    feature={}
    for layer in layers:
        feature_map[layer]=activation[layer].requires_grad_(True)
        bhook_name='feature_map_'+layer
        bhook=feature_map[layer].requires_grad_(True).register_hook(getGrad(bhook_name))
        bhooks.append(bhook)
        if 'vit' in feature_network or 'deit' in feature_network:
            if 'clip' in feature_network:
                feature_post[layer]=feature_map[layer][1:,:,:].permute(1,0,2)
            else:
                feature_post[layer]=feature_map[layer][:,1:,:]
            feature[layer] = feature_post[layer].mean(dim=1)
        elif 'swin' in feature_network or 'resmlp' in feature_network or 'mixer' in feature_network or 'gmlp' in feature_network or 'rmlp' in feature_network:
            feature[layer] = feature_map[layer].mean(dim=1)
        else:
            feature[layer] = feature_map[layer].mean(dim=[2,3])
    
    N=num_gen
    
    fid_res={}
    kid_res={}
    cka_res={}
    res={}
    feature_real={}
    feature_gen={}
    feature_real_group = {}
    feature_gen_group = {}
    feature_add_group = {}
    cka_res_fusion = {}
    whole_gen_feature = {}
    gen_feature = {}
    for layer in layers:
        if metrics == 'fid':
            #load parameters
            name=f"_{resolution}_before_softmax.txt"
            #dir=stats_path+'/mu_sigma/'+feature_network+'/'+layer+'/'
            dir=os.path.join(stats_path,'mu_sigma')
            model_path=os.path.join(dir,feature_network)
            path=os.path.join(model_path,layer)
            mu_real=torch.tensor(np.loadtxt(path+'/mu_real'+name)).to(device)
            sigma_real=torch.tensor(np.loadtxt(path+"/sigma_real"+name)).to(device)
            mu_gen=torch.tensor(np.loadtxt(path+"/mu_gen"+name)).to(device)
            sigma_gen=torch.tensor(np.loadtxt(path+"/sigma_gen"+name)).to(device)

            #calculate
            fid_res[layer]=fid_cal(N,mu_real,mu_gen,sigma_real,sigma_gen,feature[layer])
            print(f'fid_res({layer}): {fid_res[layer]}')
            fid_res[layer].backward()
            res[layer]=fid_res[layer]
        elif metrics == 'kid':
            #load parameters
            #dir=stats_path+'/features/'+feature_network+'/'+layer+'/'
            dir=os.path.join(stats_path,'features')
            model_path=os.path.join(dir,feature_network)
            path=os.path.join(model_path,layer)
            filename1 = path+'/feature_real.pickle'
            with open(filename1,'rb') as fi1:
                pre_feature_real=pickle.load(fi1,encoding='bytes')
                fi1.close()
            filename2 = path+'/feature_gen.pickle'
            with open(filename2,'rb') as fi2:
                pre_feature_gen=pickle.load(fi2,encoding='bytes')
                fi2.close()

            feature_real_all=pre_feature_real
            #calculate
            kid_res[layer]=kid_cal(real_features=torch.from_numpy(feature_real_all).to(device), gen_features=torch.from_numpy(pre_feature_gen).to(device), feature=feature[layer], num_subsets=50, max_subset_size=5000)
            print(f'kid_res({layer}): {kid_res[layer]}')
            kid_res[layer].backward()
            res[layer]=kid_res[layer]
        elif metrics == 'cka':
            #load parameters
            #dir=stats_path+'/features/'+feature_network+'/'+layer+'/'
            dir=os.path.join(stats_path,'features')
            model_path=os.path.join(dir,feature_network)
            path=os.path.join(model_path,layer)
            filename1 = path+'/feature_real.pickle'
            with open(filename1,'rb') as fi1:
                pre_feature_real=torch.tensor(pickle.load(fi1,encoding='bytes'))
                fi1.close()
            filename2 = path+'/feature_gen.pickle'
            with open(filename2,'rb') as fi2:
                pre_feature_gen=torch.tensor(pickle.load(fi2,encoding='bytes'))
                fi2.close()
            feature_real_all=pre_feature_real
            feature_real[layer]=feature_real_all
            feature_gen[layer]=pre_feature_gen
            #whole_gen_feature[layer]=torch.cat((feature_gen[layer],feature[layer]), axis=0)
            #calculate
            cka_res[layer]=cka_cal(real_features=feature_real[layer].to(device), gen_features=feature_gen[layer].to(device), feature=feature[layer], num_subsets=50, max_subset_size=5000, cka_normalize='2d', dimension='C', kernel='rbf', fusion_softmax_order='post_fusion')
            print(f'cka_res({layer}): {cka_res[layer]}')
            cka_res[layer].backward()    
            #res[layer]=cka_res[layer]


    '''
    torch.autograd.set_detect_anomaly(True)
    if fusion_softmax_order == 'pre_fusion':
        for layer in layers:
            feature_real[layer] = softmax3d(feature_real[layer], softmax_mode='2d')
            feature_gen[layer] = softmax3d(feature_gen[layer], softmax_mode='2d')
            feature[layer] = softmax3d(feature[layer], softmax_mode='2d').clone().requires_grad_(True)
    #Multi-level
    if metrics == 'cka':
        if groups > 0:
            for group in range(groups):
                if group == groups - 1:
                    feature_real_group[group] = fusion_features(layers=layers, features=feature_real, group=group, group_index=group_index, fusion_ways=fusion_ways, fusion_softmax_order=fusion_softmax_order, random_projection=random_projection, cka_normalize='2d',last=True)
                    feature_gen_group[group]  = fusion_features(layers=layers, features=feature_gen, group=group, group_index=group_index, fusion_ways=fusion_ways, fusion_softmax_order=fusion_softmax_order, random_projection=random_projection, cka_normalize='2d',last=True)
                    feature_add_group[group]  = fusion_features(layers=layers, features=feature, group=group, group_index=group_index, fusion_ways=fusion_ways, fusion_softmax_order=fusion_softmax_order, random_projection=random_projection, cka_normalize='2d',last=True)
                else:
                    feature_real_group[group] = fusion_features(layers=layers, features=feature_real, group=group, group_index=group_index, fusion_ways=fusion_ways, fusion_softmax_order=fusion_softmax_order, random_projection=random_projection, cka_normalize='2d',last=False)
                    feature_gen_group[group]  = fusion_features(layers=layers, features=feature_gen, group=group, group_index=group_index, fusion_ways=fusion_ways, fusion_softmax_order=fusion_softmax_order, random_projection=random_projection, cka_normalize='2d',last=False)
                    feature_add_group[group]  = fusion_features(layers=layers, features=feature, group=group, group_index=group_index, fusion_ways=fusion_ways, fusion_softmax_order=fusion_softmax_order, random_projection=random_projection, cka_normalize='2d',last=False)
                print(feature_real_group[group].shape)
                if random_projection:
                    feature_gen_group[group] = random_projection(feature_gen_group[group], out_channels=768).detach().cpu().numpy()
                    feature_real_group[group] = random_projection(feature_real_group[group], out_channels=768).detach().cpu().numpy()
                    feature_add_group[group] = random_projection(feature_add_group[group], out_channels=768).detach().cpu().numpy()
                    print(feature_real_group[group].shape)
                #feature_addgen_group[group]=feature_addgen_group[group].requires_grad_(True)
                #bhook_name_group='feature_group_'+str(group)
                #bhook=feature_addgen_group[group].requires_grad_(True).register_hook(getGrad(bhook_name_group))
                #bhooks.append(bhook)
                cka_res_fusion[group] = cka_cal(real_features=feature_real_group[group].to(device), gen_features=feature_gen_group[group].to(device), feature=feature_add_group[group], num_subsets=50, max_subset_size=5000, cka_normalize='2d', dimension='C', kernel='rbf', fusion_softmax_order=fusion_softmax_order)
                if not fuse_all:
                    cka_res_fusion[group].backward() 
                print(f'cka_res_fusion({group}): {cka_res_fusion[group]}')
               

        if fuse_all:
            if fusion_ways == 'sum':
                feature_real_fuse_all = feature_real_group[0]
                feature_gen_fuse_all = feature_gen_group[0]
                feature_add_fuse_all = feature_add_group[0]
                for i in range(1, len(feature_gen_group)):
                    feature_real_fuse_all = torch.cat((feature_real_fuse_all, feature_real_group[i]), dim=1)
                    feature_gen_fuse_all  = torch.cat((feature_gen_fuse_all, feature_gen_group[i]), dim=1)
                    feature_add_fuse_all  = torch.cat((feature_add_fuse_all, feature_add_group[i]), dim=1)
                cka_res_fuse_all = cka_cal(real_features=feature_real_fuse_all.to(device), gen_features=feature_gen_fuse_all.to(device), feature=feature_add_fuse_all, num_subsets=50, max_subset_size=5000, cka_normalize='2d', dimension='C', kernel='rbf', fusion_softmax_order=fusion_softmax_order)
                print(f'cka_res_fuse_all: {cka_res_fuse_all}')
            else:
                for i in range(0, len(feature_gen_group)):
                    feature_real_group[i] = random_projection(feature_real_group[i], out_channels=768).detach().cpu()
                    feature_gen_group[i]  = random_projection(feature_gen_group[i], out_channels=768).detach().cpu()
                    feature_add_group[i]  = random_projection(feature_add_group[i], out_channels=768).detach().cpu()
                fuse_all_feature_real = feature_real_group[0]
                fuse_all_feature_gen  = feature_gen_group[0]
                fuse_all_feature_add = feature_add_group[0]
                if fuse_all_ways == 'cat':
                    for i in range(1, len(feature_gen_group)):
                        fuse_all_feature_real = torch.cat((fuse_all_feature_real, feature_real_group[i]), dim=1)
                        fuse_all_feature_gen  = torch.cat((fuse_all_feature_gen, feature_gen_group[i]), dim=1)
                        fuse_all_feature_add  = torch.cat((fuse_all_feature_add, feature_add_group[i]), dim=1)
                else:
                    for i in range(1, len(feature_gen_group)):
                        fuse_all_feature_real += feature_real_group[i]
                        fuse_all_feature_gen  += feature_gen_group[i]
                        fuse_all_feature_add  += feature_add_group[i]
                cka_res_fuse_all = cka_cal(real_features=fuse_all_feature_real.to(device), gen_features=fuse_all_feature_gen.to(device), feature=fuse_all_feature_add, num_subsets=50, max_subset_size=5000, cka_normalize='2d', dimension='C', kernel='rbf', fusion_softmax_order=fusion_softmax_order)
                print(f'cka_res_fuse_all: {cka_res_fuse_all}')
            #cka_res_fuse_all.backward()
    '''
    #grads
    cams={}
    return_maps={}
    for layer in layers:
        bhook_name='feature_map_'+layer
        grads=grad_block[bhook_name]
        if 'vit' in feature_network or 'deit' in feature_network:
            if 'clip' in feature_network: 
                alpha = torch.mean(torch.pow(grads[1:,:,:],2), axis=0,keepdim=True)
                heatmap = alpha * feature_map[layer][1:,:,:]
                h_w=int(math.sqrt(int(heatmap.shape[0])))
                heatmap=rearrange(heatmap,'(h w) b c -> b c h w', h=h_w)
            elif 'moco' in feature_network:
                alpha = torch.mean(torch.pow(grads[:,1:,:],2), axis=1,keepdim=True)
                heatmap = alpha * feature_map[layer][:,1:,:]
                h_w=int(math.sqrt(int(heatmap.shape[1])))
                heatmap=rearrange(heatmap,'b (h w) c -> b c h w', h=h_w)
            else:
                alpha = torch.mean(torch.pow(grads[:,1:,:],2), axis=1,keepdim=True)
                heatmap = alpha * feature_map[layer][:,1:,:]
                h_w=int(math.sqrt(int(heatmap.shape[1])))
                heatmap=rearrange(heatmap,'b (h w) c -> b c h w', h=h_w)
        elif 'swin' in feature_network or 'resmlp' in feature_network or 'mixer' in feature_network or 'gmlp' in feature_network or 'rmlp' in feature_network:
            alpha = torch.mean(torch.pow(grads,2), axis=1,keepdim=True)
            heatmap = alpha * feature_map[layer]
            h_w=int(math.sqrt(int(heatmap.shape[1])))
            heatmap=rearrange(heatmap,'b (h w) c -> b c h w', h=h_w)
        else:
            alpha = torch.mean(torch.pow(grads,2), axis=(2,3),keepdim=True)
            heatmap = alpha * feature_map[layer]
        heatmap = heatmap.sum(dim=1)
        heatmap=heatmap[0].detach().cpu().numpy().astype(np.float32)
        max_heatmap = heatmap.max()
        min_heatmap = heatmap.min()
        heatmap = (heatmap - min_heatmap) / (max_heatmap - min_heatmap)
        heatmap = np.clip((heatmap*255.0).astype(np.uint8),0.0,255.0)
        heatmap = np.array(Image.fromarray(heatmap).resize(image.size, resample=Image.LANCZOS).convert('L'))
        return_map = heatmap
        overlay_image = show_heatmap_on_image(image=np.array(image),sensitivity_map=heatmap)
        print(overlay_image.shape)
        cam = Image.fromarray(overlay_image.transpose(1, 2, 0))
        return_maps[layer]=return_map
        cams[layer]=cam

    for i in range(len(fhooks)):
        fhooks[i].remove()
        bhooks[i].remove()
    return cams,return_maps

def image_add(image1,image2):
    img1=np.array(image1)
    img2=np.array(image2)
    add=np.maximum(img1,img2)
    return add
    

def one_line(image_file,detectors,stats_path,num_gen,outdir,metrics,groups,group_index,fusion_ways,fusion_softmax_order,random_projection,fuse_all,fuse_all_ways):
    image=Image.open(image_file)
    image=np.asarray(image)
    resolution=np.shape(image)[0]
    full_cams=[]
    return_results=[]
    number=re.sub("\D","",image_file)
    for model in detectors:
        print("Using feature_network:  "+model)
        dir=outdir
        model_path=os.path.join(dir,model)
        os.makedirs(model_path, exist_ok=True)
        savename=dir+'/'+model+'/camcam'+str(number)+'.png'
        # if os.path.exists(savename):
        #     exist_image=Image.open(savename)
        #     whole_cam=cv2.cvtColor(np.array(exist_image),cv2.COLOR_BGR2RGB)
        # else:
        whole_cam=image
        merge_heatmap=np.zeros((resolution,resolution),np.uint8)
        if model=='inception':
            layers = layer2
        elif 'vit' in model and 'clip' not in model:
            layers = layer3
        elif 'deit' in model:
            layers = layer3
        elif 'rcnn_r50' in model:
            layers = layer4
        elif 'vgg19' in model:
            layers = layer5
        elif 'clip_vit' in model:
            layers = layer6
        elif 'convnext' in model:
            layers = layer8
        elif 'repvgg' in model:
            layers = layer11
        elif 'resmlp' in model:
            layers = layer9
        elif 'swin' in model:
            layers = layer10
        elif 'mixer' in model:
            layers = layer3
        elif 'gmlp' in model:
            layers = layergmlp
        elif 'rmlp' in model:
            layers = layer11
        else:
            layers = layer1
        return_res={}
        
        cams,heatmaps=grad_cam(image_file,resolution,model,layers,stats_path,num_gen,metrics,groups,group_index,fusion_ways,fusion_softmax_order,random_projection,fuse_all,fuse_all_ways)
        #return_res=res
        for layer in layers:
            layer_path=os.path.join(model_path,layer)
            os.makedirs(layer_path, exist_ok=True)
            save_figure_name=layer_path+'/camcam'+str(number)+'.png'
            cams[layer].save(save_figure_name)
            whole_cam = np.concatenate((whole_cam, np.array(cams[layer])), axis = 1)
            merge_heatmap=image_add(merge_heatmap,heatmaps[layer])
        
        merge_path=os.path.join(model_path,'merge')
        os.makedirs(merge_path, exist_ok=True)
        save_figure_name=merge_path+'/camcam'+str(number)+'.png'
        overlay_merge_image = show_heatmap_on_image(image=image ,sensitivity_map=merge_heatmap)
        merge_cam = Image.fromarray(overlay_merge_image.transpose(1, 2, 0))
        merge_cam.save(save_figure_name)
        
        whole_cam = np.concatenate((whole_cam, merge_cam), axis = 1)
        cv2.imwrite(savename,whole_cam)
            #return_results.append(return_res)
        #plt.imshow(whole_cam)
        #plt.show()
        full_cams.append(whole_cam)
    #savetxt_name=dir+'/results'+str(number)+'.txt' 
    #if os.path.exists(savetxt_name):
        #return_results=np.loadtxt(savetxt_name)
    #else:
        #np.savetxt( savetxt_name,np.array(return_results))
    return full_cams

def txt(model):
    t=model+'(origin/'
    if model=='inception':
        layers = layer2
    elif 'vit' in model and 'clip' not in model:
        layers = layer3
    elif 'deit' in model:
        layers = layer3
    elif 'rcnn_r50' in model:
        layers = layer4
    elif 'vgg19' in model:
        layers = layer5
    elif 'clip_vit' in model:
        if 'B' in model:
            layers=layer6
        elif 'L' in model:
            layers=layer7
    elif 'convnext' in model or 'repvgg' in model:
        layers = layer8
    elif 'resmlp' in model:
        layers = layer9
    elif 'swin' in model:
        layers = layer10
    else:
        layers = layer1
    
    for layer in layers:
        t+=layer
        t+='/'
    t+='Mix)'
    return t
    
@click.command()
@click.pass_context
@click.option('--detectors', help='Choose detectors to get heatmaps', type=parse_comma_separated_list, default='inception', metavar='STRING', required=True, show_default=True)
@click.option('--generate_image_path', help='Generated images to get heatmaps', type=str, default=None, metavar='STRING', required=True, show_default=True)
@click.option('--stats_path', help='Stats path', type=str, default=None, metavar='STRING', required=True, show_default=True)
@click.option('--html_path', help='Html path', type=str, default='./visual_html', metavar='STRING', show_default=True)
@click.option('--html_name', help='Html name', type=str, default='visualize_grad_cam', metavar='STRING', show_default=True)
@click.option('--num_gen', help='Number of images to generate for computation when generate dataset is none', type=int, default=50000, metavar='INT', show_default=True)
@click.option('--metrics', help='Choose metrics', type=click.Choice(['fid', 'kid','cka']), default='fid', show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, default=None, metavar='STRING', required=True)
# multi-level features fusion
@click.option('--groups', help='How many groups combined in various layers', type=int, default=3, metavar='INT', show_default=True)
@click.option('--group_index', help='The layer index for grouping different layers', type=parse_comma_separated_list, default='11', metavar='STRING', show_default=True)
@click.option('--fusion_ways', help='How to combine multi-level features: concat or sum', type=str, default=None, metavar='STRING', show_default=True)
@click.option('--fusion_softmax_order', help='The order of softmax for fusing features, `pre_fusion` or `post_fusion` ', type=str, default='pre_fusion', metavar='STRING', show_default=True)
# random projection
@click.option('--random_projection', help='Perform random projection for more efficient computation', type=bool, default=False, metavar='BOOL', show_default=True)
@click.option('--fuse_all', help='Fuse low, mid, high-level features', type=bool, default=False, metavar='BOOL', show_default=True)
@click.option('--fuse_all_ways', help='How to fusing multi-level features: concat or sum', type=str, default=None, metavar='STRING', show_default=True)
def grad_cam_html(ctx,detectors,generate_image_path,stats_path,html_path,html_name,num_gen,metrics,outdir,groups,group_index,fusion_ways,fusion_softmax_order,random_projection,fuse_all,fuse_all_ways):
    new_generated_dataset_url=generate_image_path
    files=os.listdir(new_generated_dataset_url)
    files.sort()
    numbers=len(files)
    cams=[]
    results=[]
    for f in files:
        image_file=os.path.join(new_generated_dataset_url,f)
        print(image_file)
        cam=one_line(image_file,detectors,stats_path,num_gen,outdir,metrics,groups,group_index,fusion_ways,fusion_softmax_order,random_projection,fuse_all,fuse_all_ways)
        cams.append(cam)
    #html_visualizer
    num_rows=numbers
    num_cols=len(detectors)
    html = HtmlPageVisualizer(num_rows, num_cols)
    html.set_headers(detectors)
    for i in range(num_rows):
        for j in range(num_cols):
            html.set_cell(i, j, text=txt(detectors[j]), image=cams[i][j], highlight=False)
            #html.set_cell(i, j, text=txt(detectors[j],results[i][j]), image=cams[i][j], highlight=False)
    os.makedirs(html_path, exist_ok=True)
    html.save(html_path+'/'+html_name+'.html')

if __name__ == "__main__":
    grad_cam_html()
