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
import numpy as np

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
    L_X = np.dot(X.T, X)
    L_Y = np.dot(Y.T, Y)
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
    if opts.transport_sample:
        real_features = real_features.transpose(1, 0)
        gen_features = gen_features.transpose(1, 0)
    m = min(min(real_features.shape[0], gen_features.shape[0]), max_subset_size)
    cka = 0
    for _subset_idx in range(num_subsets):
        x = gen_features[np.random.choice(gen_features.shape[0], m, replace=False)]
        y = real_features[np.random.choice(real_features.shape[0], m, replace=False)]
        if opts.kernel == 'rbf':
            cka_s = kernel_CKA(x, y, sigma=opts.sigma)
        elif opts.kernel == 'poly':
            cka_s = poly_kernel_CKA(x, y)
        else:
            cka_s = linear_CKA(x, y)
        cka += cka_s
    cka = cka / m
    return float(cka)
   # if opts.kernel:
   #     cka = kernel_CKA(real_features, gen_features, sigma=opts.sigma)
   # else:
   #     cka = linear_CKA(real_features, gen_features)
   # return float(cka)

def compute_cka(opts, max_real, num_gen, num_subsets, max_subset_size, detector_url=None):
    if detector_url is None:
        detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True)  # Return raw features before the softmax layer.

    if opts.rank != 0:
        return float('nan')


    cka_res = {}

    # real_dataset
    res_real = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, batch_size=opts.eval_bs, capture_all=True, max_items=max_real, layers=opts.layers)

    # fake_dataset
    if opts.generate is not None:
        res_gen = metric_utils.compute_feature_stats_for_generate_dataset(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=1, batch_size=opts.eval_bs, capture_all=True, max_items=num_gen, layers=opts.layers)
    else:
        res_gen = metric_utils.compute_feature_stats_for_generator(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=1, batch_size=opts.eval_bs, capture_all=True, max_items=num_gen, layers=opts.layers)
    # caculate
    for layer in opts.layers:
        cka_res[layer] = cka_cal(res_real[layer].get_all(), res_gen[layer].get_all(), max_subset_size, num_subsets,
                                 opts)
    return cka_res
