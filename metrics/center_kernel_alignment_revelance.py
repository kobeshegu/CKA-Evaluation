# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Kernel Inception Distance (KID) from the paper "Demystifying MMD
GANs". Matches the original implementation by Binkowski et al. at
https://github.com/mbinkowski/MMD-GAN/blob/master/gan/compute_scores.py"""


from . import metric_utils
# ----------------------------------------------------------------------------
import math
import pickle
import numpy as np
import torch
import os
import torch.nn as nn
import time
import random
import cv2


detector_urls=dict(
    swav = './detector/swav.pth.tar',
    clip = './detector/clip.pt',
    spr = './detector/spr.pth.tar',
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
    clip_vit_B16 = './detector/clip_vit_B16.pt',
    clip_vit_B32 = './detector/clip_vit_B32.pt',
    clip_vit_L14 = './detector/clip_vit_L14.pt',
    convnext_base = '/mnt/petrelfs/yangmengping/ckpt/convnext/convnext_base_1k_224_ema.pth'

)

models=["inception","clip_vit_B32","moco_vit_i","convnext_base","vitcls_base_patch16_224"]
# layer1=["Conv2d_4a_3x3","Mixed_5d","Mixed_6e","Mixed_7c"]
# layer2=["blocks.0","blocks.1","blocks.2","blocks.3","blocks.4","blocks.5","blocks.6","blocks.7","blocks.8","blocks.9","blocks.10","blocks.11",]
# layer3=["transformer.resblocks.0","transformer.resblocks.1","transformer.resblocks.2","transformer.resblocks.3","transformer.resblocks.4","transformer.resblocks.5","transformer.resblocks.6","transformer.resblocks.7","transformer.resblocks.8","transformer.resblocks.9","transformer.resblocks.10","transformer.resblocks.11"]
# layer4=["stages.0","stages.1","stages.2","stages.3"]
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
layer12 = ["layer1.0","layer2.0","layer3.0","layer4.0"]


def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    #return np.dot(np.dot(H, K), H)
    return np.dot(K, H)  # KH

def rbf(GX, sigma=None):
    #GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = np.exp(KX)
    return KX

def poly(GX, poly_constant=1, poly_power=3):
    return (poly_constant + np.dot(GX, GX.T)) ** poly_power

def poly_Kernel_HSIC(X, Y):
    L_X = np.dot(X.T, X)
    L_Y = np.dot(Y.T, Y)
    return np.sum(centering(poly(L_X)) * centering(poly(L_Y)))
    

def kernel_HSIC(X, Y, sigma):
    L_X = np.dot(X.T, X)
    L_Y = np.dot(Y.T, Y)
    return np.sum(centering(rbf(L_X, sigma)) * centering(rbf(L_Y, sigma)))


def linear_HSIC(X, Y):
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    return np.sum(centering(L_X) * centering(L_Y))


def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = np.sqrt(linear_HSIC(X, X))
    var2 = np.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)

def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = np.sqrt(kernel_HSIC(X, X, sigma))
    var2 = np.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)

def poly_kernel_CKA(X, Y):
    hsic = poly_Kernel_HSIC(X, Y)
    var1 = np.sqrt(poly_Kernel_HSIC(X, X))
    var2 = np.sqrt(poly_Kernel_HSIC(Y, Y))

    return hsic / (var1 * var2)

def cka_cal(real_features, gen_features, max_subset_size, num_subsets, opts):
    if opts.rank != 0:
        return float('nan')
    print(gen_features.shape)
    print(real_features.shape)
    #if opts.transport_sample:
        #real_features = real_features.transpose(1, 0)
        #gen_features = gen_features.transpose(1, 0)
    m = min(min(real_features.shape[0], gen_features.shape[0]), max_subset_size)
    cka = 0
    x = gen_features.transpose(1, 0)
    y = real_features.transpose(1, 0)
    if opts.kernel == 'rbf':
        cka_s = kernel_CKA(x, y, sigma=opts.sigma)
    elif opts.kernel == 'poly':
        cka_s = poly_kernel_CKA(x, y)
    elif opts.kernel == 'linear':
        cka_s = linear_CKA(x, y)
    cka += cka_s
    cka = cka / num_subsets
    return float(cka)
   # if opts.kernel:
   #     cka = kernel_CKA(real_features, gen_features, sigma=opts.sigma)
   # else:
   #     cka = linear_CKA(real_features, gen_features)
   # return float(cka)

def layer_get(model):

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
    elif 'spr' in model:
        layers = layer12
    else:
        layers = layer1

    # if model=='inception':
    #     layers=layer1
    # elif 'vit' in model and 'clip' not in model:
    #     layers=layer2
    # elif 'clip_vit' in model:
    #     layers=layer3
    # elif 'convnext' in model:
        # layers = layer4
    
    return layers

def compute_cka(opts):
    #if detector_url is None:
        #detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True)  # Return raw features before the softmax layer.

    if opts.rank != 0:
        return float('nan')

    # if feature_network_f is None:
    #     model_f='inception'
    # elif feature_network_f == 'spr':
    #     model_f='resnet50'
    # else:
    #     model_f=feature_network_f

    # if feature_network_h is None:
    #     model_h='inception'
    # elif feature_network_h == 'spr':
    #     model_h='resnet50'
    # else:
    #     model_h=feature_network_h
    
    layers_1=layer_get(opts.detector1)
    layers_2=layer_get(opts.detector2)
    if opts.detector1 in detector_urls.keys():
        url1 = detector_urls[opts.detector1]
    else:
        url1 = None
    if opts.detector2 in detector_urls.keys():
        url2 = detector_urls[opts.detector2]
    else:
        url2 = None

    # detector1
    opts.feature_network = opts.detector1
    res_f = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=url1, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, batch_size=opts.eval_bs, capture_all=True, max_items=opts.max_real, layers=layers_1)
    # dataset_h
    opts.feature_network = opts.detector2
    res_h = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=url2, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, batch_size=opts.eval_bs, capture_all=True, max_items=opts.max_real, layers=layers_2)
    # caculate
    ##### here we only calculate the last layers
    cka_res = cka_cal(res_f[layers_1[len(layers_1)-1]].get_all(), res_h[layers_2[len(layers_2)-1]].get_all(), 10000, 1, opts)

    os.makedirs(opts.save_res, exist_ok=True)
    output_name = opts.save_res  + '/' + 'cka_'+opts.save_name+'.txt'
    f = open(output_name,'a')
    f.write("-----------new-metrics------------\n")
    rew='F:'+opts.detector1+' H:'+opts.detector2+'_'+':'+str(cka_res)+'\n'
    f.write(rew)
    f.close()    

    return cka_res
