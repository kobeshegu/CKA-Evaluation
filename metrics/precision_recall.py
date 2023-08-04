# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Precision/Recall (PR) from the paper "Improved Precision and Recall
Metric for Assessing Generative Models". Matches the original implementation
by Kynkaanniemi et al. at
https://github.com/kynkaat/improved-precision-and-recall-metric/blob/master/precision_recall.py"""

import torch
from . import metric_utils
import os

#----------------------------------------------------------------------------

def compute_distances(row_features, col_features, num_gpus, rank, col_batch_size):
    assert 0 <= rank < num_gpus
    num_cols = col_features.shape[0]
    num_batches = ((num_cols - 1) // col_batch_size // num_gpus + 1) * num_gpus
    col_batches = torch.nn.functional.pad(col_features, [0, 0, 0, -num_cols % num_batches]).chunk(num_batches)
    dist_batches = []
    for col_batch in col_batches[rank :: num_gpus]:
        dist_batch = torch.cdist(row_features.unsqueeze(0), col_batch.unsqueeze(0))[0]
        for src in range(num_gpus):
            dist_broadcast = dist_batch.clone()
            if num_gpus > 1:
                torch.distributed.broadcast(dist_broadcast, src=src)
            dist_batches.append(dist_broadcast.cpu() if rank == 0 else None)
    return torch.cat(dist_batches, dim=1)[:, :num_cols] if rank == 0 else None

#----------------------------------------------------------------------------

def compute_pr(opts, nhood_size, row_batch_size, col_batch_size, detector_url=None):

    if detector_url is None:
        detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/vgg16.pkl'
    detector_kwargs = dict(return_features=True)

    if opts.feature_network is None:
        model = 'inception'
    else:
        model = opts.feature_network

    os.makedirs(opts.save_res, exist_ok=True)
    output_name = opts.save_res  + '/' + 'pr_'+opts.save_name+'.txt'
    f = open(output_name,'a')
    f.write("-----------new-metrics------------\n")
 
    # real_dataset
    res_real = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_all=True, max_items=opts.max_real, layers=opts.layers)

    # generate_dataset
    if opts.generate is not None:
        res_gen = metric_utils.compute_feature_stats_for_generate_dataset(
                opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
                rel_lo=0, rel_hi=1, batch_size=opts.eval_bs, capture_all=True, max_items=opts.num_gen,layers=opts.layers)
    else:
         res_gen = metric_utils.compute_feature_stats_for_generator(
                opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
                rel_lo=0, rel_hi=1, batch_size=opts.eval_bs, capture_all=True, max_items=opts.num_gen,layers=opts.layers)
    results = dict()
    for layer in opts.layers:
        real_features = res_real[layer].get_all_torch().to(torch.float16).to(opts.device)
        gen_features = res_gen[layer].get_all_torch().to(torch.float16).to(opts.device)
        for name, manifold, probes in [('precision', real_features, gen_features), ('recall', gen_features, real_features)]:
            kth = []
            for manifold_batch in manifold.split(row_batch_size):
                dist = compute_distances(row_features=manifold_batch, col_features=manifold, num_gpus=opts.num_gpus, rank=opts.rank, col_batch_size=col_batch_size)
                kth.append(dist.to(torch.float32).kthvalue(nhood_size + 1).values.to(torch.float16) if opts.rank == 0 else None)
            kth = torch.cat(kth) if opts.rank == 0 else None
            pred = []
            for probes_batch in probes.split(row_batch_size):
                dist = compute_distances(row_features=probes_batch, col_features=manifold, num_gpus=opts.num_gpus, rank=opts.rank, col_batch_size=col_batch_size)
                pred.append((dist <= kth).any(dim=1) if opts.rank == 0 else None)
            results[name + layer] = float(torch.cat(pred).to(torch.float32).mean() if opts.rank == 0 else 'nan')
        save_res = model + '_' + layer + ':'+ 'Precision:' + str(results['precision' + layer]) + 'Recall:' + str(results['recall' + layer]) + '\n'
        f.write(save_res)   
    f.close()
    return results 

#----------------------------------------------------------------------------
