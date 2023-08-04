# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Frechet Inception Distance (FID) from the paper
"GANs trained by a two time-scale update rule converge to a local Nash
equilibrium". Matches the original implementation by Heusel et al. at
https://github.com/bioinf-jku/TTUR/blob/master/fid.py"""

from http.client import REQUESTED_RANGE_NOT_SATISFIABLE
import numpy as np
import scipy.linalg
from . import metric_utils
import os

#----------------------------------------------------------------------------

def fid_cal(mu_real, sigma_real, mu_gen, sigma_gen,opts):
    if opts.rank != 0:
        return float('nan')
    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid_res = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid_res)

def fake_calculate(opts, f, res_real, sfid, rfid, detector_url,detector_kwargs):
    fid_res={}
    num_gen=opts.num_gen
    #real_dataset
    mu_real={}
    sigma_real={}
    f.write("--------new-epoch--------\n")
    #generation
    mu_gen={}
    sigma_gen={}

    if opts.generate is not None:
        res_gen= metric_utils.compute_feature_stats_for_generate_dataset(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=1, batch_size=opts.eval_bs, capture_mean_cov=True, capture_all=True, max_items=opts.random_size,layers=opts.layers)
    else:
        res_gen= metric_utils.compute_feature_stats_for_generator(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=1, batch_size=opts.eval_bs, capture_mean_cov=True, capture_all=True, max_items=opts.random_size,layers=opts.layers)

    #caculate
    for layer in opts.layers:
        mu_real[layer],sigma_real[layer]=res_real[layer].get_mean_cov()
        mu_gen[layer],sigma_gen[layer]=res_gen[layer].get_mean_cov()
        #print(res_gen[layer].shape)
        if opts.feature_network is None:
            model='inception'
        elif opts.feature_network == 'spr':
            model='resnet50'
        else:
            model=opts.feature_network
        fid_res[layer]=fid_cal(mu_real[layer], sigma_real[layer], mu_gen[layer], sigma_gen[layer],opts)
        rew=model+'_'+layer+':'+str(fid_res[layer])+'\n'
        f.write(rew)
    return fid_res

def compute_fid(opts, sfid=False, rfid=False, detector_url=None):
    
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
 
    fid_res={}
    os.makedirs(opts.save_res, exist_ok=True)
    output_name = opts.save_res  + '/' + 'fid_'+opts.save_name+'.txt'
    f=open(output_name,'a')
    f.write("-----------new-metrics------------\n")
    
    #real_dataset
    mu_real={}
    sigma_real={}
    
    res_real = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, batch_size=opts.eval_bs, capture_mean_cov=True, capture_all=True, max_items=opts.max_real, layers=opts.layers)
    
    #generate_dataset & calculate
    if opts.random == True:
        fid_list=[]
        fid_mean={}
        fid_std={}
        f1_name = opts.save_res  + '/' + 'fid_mean_'+opts.save_name+'.txt'
        f2_name = opts.save_res  + '/' + 'fid_std_'+opts.save_name+'.txt'
        f1=open(f1_name,'a')
        f1.write("--------new-metrics----------\n")
        f2=open(f2_name,'a')
        f2.write("--------new-metrics----------\n")
        for num in range(opts.random_num):
            print('Epoch: %d' %num)
            fid_list.append(fake_calculate(opts, f, res_real, sfid, rfid, detector_url,detector_kwargs))
        for layer in opts.layers:
            res=[]
            for num in range(opts.random_num):
                res.append(fid_list[num][layer])
            res=np.array(res)
            fid_mean[layer]=np.mean(res)
            fid_std[layer]=np.std(res)
            f1_res=model+'_'+layer+':'+str(fid_mean[layer])+'\n'
            f1.write(f1_res)
            f2_res=model+'_'+layer+':'+str(fid_std[layer])+'\n'
            f2.write(f2_res)
        f1.close()
        f2.close()
        f.close()
        return fid_mean
    else:
        mu_gen={}
        sigma_gen={}
        if opts.generate is not None:
            res_gen= metric_utils.compute_feature_stats_for_generate_dataset(
                opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
                rel_lo=0, rel_hi=1, batch_size=opts.eval_bs, capture_mean_cov=True, capture_all=True, max_items=opts.num_gen,layers=opts.layers)
        else:
            res_gen= metric_utils.compute_feature_stats_for_generator(
                opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
                rel_lo=0, rel_hi=1, batch_size=opts.eval_bs, capture_mean_cov=True, capture_all=True, max_items=opts.num_gen,layers=opts.layers)
        for layer in opts.layers:
            mu_real[layer],sigma_real[layer]=res_real[layer].get_mean_cov()
            mu_gen[layer],sigma_gen[layer]=res_gen[layer].get_mean_cov()
            if opts.save_stats is not None:
                dir=os.path.join(opts.save_stats,'mu_sigma')
                model_path=os.path.join(dir,model)
                path=os.path.join(model_path,layer)
                os.makedirs(path, exist_ok=True)
                np.savetxt(path+"/mu_real_"+str(resolution)+"_before_softmax.txt", mu_real[layer])
                np.savetxt(path+"/sigma_real_"+str(resolution)+"_before_softmax.txt", sigma_real[layer])
                np.savetxt(path+"/mu_gen_"+str(resolution)+"_before_softmax.txt", mu_gen[layer])
                np.savetxt(path+"/sigma_gen_"+str(resolution)+"_before_softmax.txt", sigma_gen[layer])
            fid_res[layer]=fid_cal(mu_real[layer], sigma_real[layer], mu_gen[layer], sigma_gen[layer],opts)
            rew=model+'_'+layer+':'+str(fid_res[layer])+'\n'
            f.write(rew)
        f.close()
        return fid_res

#----------------------------------------------------------------------------
