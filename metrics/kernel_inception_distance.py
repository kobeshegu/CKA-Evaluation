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

import numpy as np
import pickle
import os
from . import metric_utils
import torch.nn as nn
import torch
import time 
#----------------------------------------------------------------------------

def kid_cal(real_features, gen_features, opts):
    if opts.rank != 0:
        return float('nan')
    
    n = real_features.shape[1]
    m = min(min(real_features.shape[0], gen_features.shape[0]), opts.max_subset_size)
    t = 0
    #one_time = time.time()#time
    for _subset_idx in range(opts.num_subsets):
        x = gen_features[np.random.choice(gen_features.shape[0], m, replace=False)]
        y = real_features[np.random.choice(real_features.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    #one_time_time = time.time() - one_time
    #print(f'Total KID is {float(t)}, Calculation time:{one_time_time} s.')
    kid = t / opts.num_subsets / m
    return float(kid)

def fake_calculate(opts, f, res_real, rfid, detector_url,detector_kwargs):
    kid_res={}
    num_gen=opts.num_gen
    #real_dataset
    feature_real={}

    #generation
    feature_gen={}
    f.write("-------new-epoch--------\n")
    if opts.generate is not None:
        res_gen= metric_utils.compute_feature_stats_for_generate_dataset(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=1, batch_size=opts.eval_bs, capture_all=True, capture_mean_cov=True, max_items=opts.random_size,layers=opts.layers)
    else:
        res_gen= metric_utils.compute_feature_stats_for_generator(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=1, batch_size=opts.eval_bs, capture_all=True, capture_mean_cov=True, max_items=opts.random_size,layers=opts.layers)

    #caculate
    for layer in opts.layers:
        if opts.feature_network is None:
            model='inception'
        elif opts.feature_network == 'spr':
            model='resnet50'
        else:
            model=opts.feature_network
        feature_real[layer]=res_real[layer].get_all()
        feature_gen[layer]=res_gen[layer].get_all()
        kid_res[layer]=kid_cal(feature_real[layer], feature_gen[layer], opts)
        rew=model+'_'+layer+':'+str(kid_res[layer])+'\n'
        f.write(rew)
    return kid_res

def compute_kid(opts, rfid=False, detector_url=None):
    resolution=opts.resolution
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    if detector_url is None:
        detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.
    if rfid:
        raise ValueError
        #detector_url = 'https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/feature_networks/inception_rand_full.pkl'
        #detector_kwargs = {}  # random inception network returns features by default
    if opts.feature_network is None:
        model='inception'
    elif opts.feature_network == 'spr':
        model='resnet50'
    else:
        model=opts.feature_network 
    os.makedirs(opts.save_res, exist_ok=True)
    output_name = opts.save_res  + '/' + 'kid_'+opts.save_name+'.txt'
    f=open(output_name,'a')
    f.write("-----------new-metrics------------\n")
    
    #real_dataset
    res_real = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, batch_size=opts.eval_bs, capture_all=True, capture_mean_cov=True, max_items=opts.max_real, layers=opts.layers)

    #generate_dataset & calculate
    if opts.random == True:
        kid_list=[]
        kid_mean={}
        kid_std={}
        f1_name = opts.save_res  + '/' + 'kid_mean_'+opts.save_name+'.txt'
        f2_name = opts.save_res  + '/' + 'kid_std_'+opts.save_name+'.txt'
        f1=open(f1_name,'a')
        f1.write("--------new-epoch----------\n")
        f2=open(f2_name,'a')
        f2.write("--------new-epoch----------\n")
        for num in range(opts.random_num):
            print('Epoch: %d' %num)
            kid_list.append(fake_calculate(opts, f, res_real, rfid, detector_url,detector_kwargs))
        for layer in opts.layers:
            res=[]
            for num in range(opts.random_num):
                res.append(kid_list[num][layer])
            res=np.array(res)
            kid_mean[layer]=np.mean(res)
            kid_std[layer]=np.std(res)
            f1_res=model+'_'+layer+':'+str(kid_mean[layer])+'\n'
            f1.write(f1_res)
            f2_res=model+'_'+layer+':'+str(kid_std[layer])+'\n'
            f2.write(f2_res)
        f1.close()
        f2.close()
        f.close()
        return kid_mean
    else:
        kid_res={}
        feature_real={}
        feature_gen={}
        if opts.generate is not None:
            res_gen= metric_utils.compute_feature_stats_for_generate_dataset(
                opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
                rel_lo=0, rel_hi=1, batch_size=opts.eval_bs, capture_all=True, capture_mean_cov=True, max_items=opts.num_gen,layers=opts.layers)
        else:
            res_gen= metric_utils.compute_feature_stats_for_generator(
                opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
                rel_lo=0, rel_hi=1, batch_size=opts.eval_bs, capture_all=True, capture_mean_cov=True, max_items=opts.num_gen,layers=opts.layers)

        for layer in opts.layers:
            feature_real[layer]=res_real[layer].get_all()
            feature_gen[layer]=res_gen[layer].get_all()
            kid_res[layer]=kid_cal(feature_real[layer], feature_gen[layer], opts)
            rew=model+'_'+layer+':'+str(kid_res[layer])+'\n'
            f.write(rew)
            if opts.save_stats is not None:
                #save features
                dir=f'{opts.save_stats}/features'
                model_path=os.path.join(dir,model)
                path=os.path.join(model_path,layer)
                os.makedirs(path, exist_ok=True)
                filename1 = path+'/feature_real.pickle'
                with open(filename1,'wb') as fo1:
                    pickle.dump(feature_real[layer],fo1,protocol = pickle.HIGHEST_PROTOCOL)
                    fo1.close()
                filename2 = path+'/feature_gen.pickle'
                with open(filename2,'wb') as fo2:
                    pickle.dump(feature_gen[layer],fo2,protocol = pickle.HIGHEST_PROTOCOL)
                    fo2.close()
        f.close()
        return kid_res
#----------------------------------------------------------------------------
