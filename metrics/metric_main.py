# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Main API for computing and reporting quality metrics."""

import os
import time
import json
import torch
import dnnlib

from . import metric_utils
from . import frechet_inception_distance
from . import kernel_inception_distance
from . import precision_recall
from . import perceptual_path_length
from . import inception_score
from . import equivariance
from . import center_kernel_alignment_torch
from . import center_kernel_alignment_numpy
from . import center_kernel_alignment_revelance
#----------------------------------------------------------------------------

_metric_dict = dict() # name => fn
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




def register_metric(fn):
    assert callable(fn)
    _metric_dict[fn.__name__] = fn
    return fn

def is_valid_metric(metric):
    return metric in _metric_dict

def list_valid_metrics():
    return list(_metric_dict.keys())

#----------------------------------------------------------------------------

def calc_metric(metric, **kwargs): # See metric_utils.MetricOptions for the full list of arguments.
    assert is_valid_metric(metric)
    opts = metric_utils.MetricOptions(**kwargs)

    # Calculate.
    start_time = time.time()
    results = _metric_dict[metric](opts)
    total_time = time.time() - start_time

    # Broadcast results.
    for key, value in list(results.items()):
        if opts.num_gpus > 1:
            for layer in opts.layers:
            	value[layer] = torch.as_tensor(value[layer], dtype=torch.float64, device=opts.device)
            	torch.distributed.broadcast(tensor=value[layer], src=0)
            	value[layer] = float(value[layer].cpu())
        results[key] = value

    # Decorate with metadata.
    return dnnlib.EasyDict(
        results         = dnnlib.EasyDict(results),
        metric          = metric,
        total_time      = total_time,
        total_time_str  = dnnlib.util.format_time(total_time),
        num_gpus        = opts.num_gpus,
    )

#----------------------------------------------------------------------------

def report_metric(result_dict, run_dir=None, snapshot_pkl=None):
    metric = result_dict['metric']
    assert is_valid_metric(metric)
    if run_dir is not None and snapshot_pkl is not None:
        snapshot_pkl = os.path.relpath(snapshot_pkl, run_dir)
    jsonl_line = json.dumps(dict(result_dict, snapshot_pkl=snapshot_pkl, timestamp=time.time()))
    print(jsonl_line)
    if run_dir is not None and os.path.isdir(run_dir):
        with open(os.path.join(run_dir, f'metric-{metric}.jsonl'), 'at') as f:
            f.write(jsonl_line + '\n')

#----------------------------------------------------------------------------
# Recommended metrics.

@register_metric
def fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    fid = frechet_inception_distance.compute_fid(opts)
    return dict(fid50k_full=fid)

@register_metric
def kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    kid = kernel_inception_distance.compute_kid(opts)
    return dict(kid50k_full=kid)

@register_metric
def pr50k3_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    precision, recall = precision_recall.compute_pr(opts, max_real=200000, nhood_size=3, row_batch_size=5000, col_batch_size=5000)
    return dict(pr50k3_full_precision=precision, pr50k3_full_recall=recall)

@register_metric
def ppl2_wend(opts):
    ppl = perceptual_path_length.compute_ppl(opts, num_samples=50000, epsilon=1e-4, space='w', sampling='end', crop=False, batch_size=2)
    return dict(ppl2_wend=ppl)

@register_metric
def eqt50k_int(opts):
    opts.G_kwargs.update(force_fp32=True)
    psnr = equivariance.compute_equivariance_metrics(opts, num_samples=50000, batch_size=4, compute_eqt_int=True)
    return dict(eqt50k_int=psnr)

@register_metric
def eqt50k_frac(opts):
    opts.G_kwargs.update(force_fp32=True)
    psnr = equivariance.compute_equivariance_metrics(opts, num_samples=50000, batch_size=4, compute_eqt_frac=True)
    return dict(eqt50k_frac=psnr)

@register_metric
def eqr50k(opts):
    opts.G_kwargs.update(force_fp32=True)
    psnr = equivariance.compute_equivariance_metrics(opts, num_samples=50000, batch_size=4, compute_eqr=True)
    return dict(eqr50k=psnr)

#----------------------------------------------------------------------------
# New fid metrics, mainly borrowed from StyleGAN-XL implementation

@register_metric
def swavfid50k_full(opts):
    # swav resnet50 as default
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'swav'
    swavfid = frechet_inception_distance.compute_fid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(swavfid50k_full=swavfid)

@register_metric
def clipfid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    # clip resnet50 as default
    opts.feature_network = 'clip'
    clipfid = frechet_inception_distance.compute_fid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(clipfid50k_full=clipfid)

@register_metric
def sprfid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'spr'
    sprfid = frechet_inception_distance.compute_fid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(sprfid50k_full=sprfid)

@register_metric
def fasterrcnn_r50_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'fasterrcnn_r50'
    fasterrcnn_r50_fid = frechet_inception_distance.compute_fid(opts)
    return dict(fasterrcnn_r50_fid50k_full=fasterrcnn_r50_fid)

@register_metric
def maskrcnn_r50_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'maskrcnn_r50'
    maskrcnn_r50_fid = frechet_inception_distance.compute_fid(opts)
    return dict(maskrcnn_r50_fid50k_full=maskrcnn_r50_fid)

@register_metric
def vggfid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'vgg'
    vggfid = frechet_inception_distance.compute_fid(opts)
    return dict(vggfid50k_full=vggfid)

@register_metric
def moco_v2_r50_i_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_v2_r50_i'
    moco_v2_r50_i_fid = frechet_inception_distance.compute_fid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(moco_v2_r50_i_fid50k_full=moco_v2_r50_i_fid)

@register_metric
def moco_r50_i_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_r50_i'
    moco_r50_i_fid = frechet_inception_distance.compute_fid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(moco_r50_i_fid50k_full=moco_r50_i_fid)

@register_metric
def moco_r50_f_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_r50_f'
    moco_r50_f_fid = frechet_inception_distance.compute_fid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(moco_r50_f_fid50k_full=moco_r50_f_fid)

@register_metric
def moco_r50_anime_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_r50_anime'
    moco_r50_anime_fid = frechet_inception_distance.compute_fid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(moco_r50_anime_fid50k_full=moco_r50_anime_fid)

@register_metric
def moco_r50_celeba_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_r50_celeba'
    moco_r50_celeba_fid = frechet_inception_distance.compute_fid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(moco_r50_celeba_fid50k_full=moco_r50_celeba_fid)

@register_metric
def moco_r50_church_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_r50_church'
    moco_r50_church_fid = frechet_inception_distance.compute_fid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(moco_r50_church_fid50k_full=moco_r50_church_fid)

@register_metric
def moco_r50_afhq_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_r50_afhq'
    moco_r50_afhq_fid = frechet_inception_distance.compute_fid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(moco_r50_afhq_fid50k_full=moco_r50_afhq_fid)

@register_metric
def moco_r50_lsunhorse_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_r50_lsunhorse'
    moco_r50_lsunhorse_fid = frechet_inception_distance.compute_fid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(moco_r50_lsunhorse_fid50k_full=moco_r50_lsunhorse_fid)

@register_metric
def moco_r50_lsunbedroom_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_r50_lsunbedroom'
    moco_r50_lsunbedroom_fid = frechet_inception_distance.compute_fid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(moco_r50_lsunbedroom_fid50k_full=moco_r50_lsunbedroom_fid)

@register_metric
def moco_r50_s0_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_r50_s0'
    moco_r50_s0_fid = frechet_inception_distance.compute_fid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(moco_r50_s0_fid50k_full=moco_r50_s0_fid)

@register_metric
def moco_r50_s1_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_r50_s1'
    moco_r50_s1_fid = frechet_inception_distance.compute_fid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(moco_r50_s1_fid50k_full=moco_r50_s1_fid)

@register_metric
def moco_r50_s2_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_r50_s2'
    mocov2_r50_s2_fid = frechet_inception_distance.compute_fid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(moco_r50_s2_fid50k_full=mocov2_r50_s2_fid)

@register_metric
def moco_vit_i_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_vit_i'
    moco_vit_i_fid = frechet_inception_distance.compute_fid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(moco_vit_i_fid50k_full=moco_vit_i_fid)

@register_metric
def moco_vit_f_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_vit_f'
    moco_vit_f_fid = frechet_inception_distance.compute_fid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(moco_vit_f_fid50k_full=moco_vit_f_fid)

@register_metric
def moco_vit_anime_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_vit_anime'
    moco_vit_anime_fid = frechet_inception_distance.compute_fid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(moco_vit_anime_fid50k_full=moco_vit_anime_fid)

@register_metric
def moco_vit_celeba_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_vit_celeba'
    moco_vit_celeba_fid = frechet_inception_distance.compute_fid(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_vit_celeba_fid50k_full=moco_vit_celeba_fid)

@register_metric
def moco_vit_church_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_vit_church'
    moco_vit_church_fid = frechet_inception_distance.compute_fid(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_vit_church_fid50k_full=moco_vit_church_fid)

@register_metric
def moco_vit_afhq_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_vit_afhq'
    moco_vit_afhq_fid = frechet_inception_distance.compute_fid(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_vit_afhq_fid50k_full=moco_vit_afhq_fid)

@register_metric
def moco_vit_lsunhorse_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_vit_lsunhorse'
    moco_vit_lsunhorse_fid = frechet_inception_distance.compute_fid(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_vit_lsunhorse_fid50k_full=moco_vit_lsunhorse_fid)

@register_metric
def moco_vit_lsunbedroom_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_vit_lsunbedroom'
    moco_vit_lsunbedroom_fid = frechet_inception_distance.compute_fid(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_vit_lsunbedroom_fid50k_full=moco_vit_lsunbedroom_fid)

@register_metric
def moco_vit_s0_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_vit_s0'
    moco_vit_s0_fid = frechet_inception_distance.compute_fid(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_vit_s0_fid50k_full=moco_vit_s0_fid)

@register_metric
def moco_vit_s1_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_vit_s1'
    moco_vit_s1_fid = frechet_inception_distance.compute_fid(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_vit_s1_fid50k_full=moco_vit_s1_fid)

@register_metric
def moco_vit_s2_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_vit_s2'
    mocov2_vit_s2_fid = frechet_inception_distance.compute_fid(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_vit_s2_fid50k_full=mocov2_vit_s2_fid)

#----------------------------------------------------------------------------
# New kid metrics, mainly borrowed from StyleGAN-XL implementation

@register_metric
def swavkid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'swav'
    swavkid = kernel_inception_distance.compute_kid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(swavkid50k_full=swavkid)

@register_metric
def clipkid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'clip'
    clipkid = kernel_inception_distance.compute_kid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(clipkid50k_full=clipkid)

@register_metric
def sprkid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'spr'
    sprkid = kernel_inception_distance.compute_kid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(sprkid50k_full=sprkid)

@register_metric
def fasterrcnn_r50_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'fasterrcnn_r50'
    fasterrcnn_r50_kid = kernel_inception_distance.compute_kid(opts)
    return dict(fasterrcnn_r50_kid50k_full=fasterrcnn_r50_kid)

@register_metric
def maskrcnn_r50_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'maskrcnn_r50'
    maskrcnn_r50_kid = kernel_inception_distance.compute_kid(opts)
    return dict(maskrcnn_r50_kid50k_full=maskrcnn_r50_kid)

@register_metric
def vggkid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'vgg'
    vggkid = kernel_inception_distance.compute_kid(opts)
    return dict(vggkid50k_full=vggkid)

@register_metric
def moco_v2_r50_i_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_v2_r50_i'
    moco_v2_r50_i_kid = kernel_inception_distance.compute_kid(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_v2_r50_i_kid50k_full=moco_v2_r50_i_kid)

@register_metric
def moco_r50_i_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_r50_i'
    moco_r50_i_kid = kernel_inception_distance.compute_kid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(moco_r50_i_kid50k_full=moco_r50_i_kid)

@register_metric
def moco_r50_f_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_r50_f'
    moco_r50_f_kid =kernel_inception_distance.compute_kid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(moco_r50_f_kid50k_full=moco_r50_f_kid)

@register_metric
def moco_r50_anime_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_r50_anime'
    moco_r50_anime_kid = kernel_inception_distance.compute_kid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(moco_r50_anime_kid50k_full=moco_r50_anime_kid)

@register_metric
def moco_r50_celeba_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_r50_celeba'
    moco_r50_celeba_kid = kernel_inception_distance.compute_kid(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_r50_celeba_kid50k_full=moco_r50_celeba_kid)

@register_metric
def moco_r50_church_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_r50_church'
    moco_r50_church_kid = kernel_inception_distance.compute_kid(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_r50_church_kid50k_full=moco_r50_church_kid)

@register_metric
def moco_r50_afhq_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_r50_afhq'
    moco_r50_afhq_kid = kernel_inception_distance.compute_kid(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_r50_afhq_kid50k_full=moco_r50_afhq_kid)

@register_metric
def moco_r50_lsunhorse_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_r50_lsunhorse'
    moco_r50_lsunhorse_kid = kernel_inception_distance.compute_kid(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_r50_lsunhorse_kid50k_full=moco_r50_lsunhorse_kid)

@register_metric
def moco_r50_lsunbedroom_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_r50_lsunbedroom'
    moco_r50_lsunbedroom_kid = kernel_inception_distance.compute_kid(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_r50_lsunbedroom_kid50k_full=moco_r50_lsunbedroom_kid)

@register_metric
def moco_r50_s0_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_r50_s0'
    moco_r50_s0_kid = kernel_inception_distance.compute_kid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(moco_r50_s0_kid50k_full=moco_r50_s0_kid)

@register_metric
def moco_r50_s1_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_r50_s1'
    moco_r50_s1_kid =kernel_inception_distance.compute_kid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(moco_r50_s1_kid50k_full=moco_r50_s1_kid)

@register_metric
def moco_r50_s2_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_r50_s2'
    mocov2_r50_s2_kid = kernel_inception_distance.compute_kid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(moco_r50_s2_kid50k_full=mocov2_r50_s2_kid)

@register_metric
def moco_vit_i_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_vit_i'
    moco_vit_i_kid = kernel_inception_distance.compute_kid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(moco_vit_i_kid50k_full=moco_vit_i_kid)

@register_metric
def moco_vit_f_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_vit_f'
    moco_vit_f_kid = kernel_inception_distance.compute_kid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(moco_vit_f_kid50k_full=moco_vit_f_kid)

@register_metric
def moco_vit_anime_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_vit_anime'
    moco_vit_anime_kid = kernel_inception_distance.compute_kid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(moco_vit_anime_kid50k_full=moco_vit_anime_kid)

@register_metric
def moco_vit_celeba_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_vit_celeba'
    moco_vit_celeba_kid = kernel_inception_distance.compute_kid(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_vit_celeba_kid50k_full=moco_vit_celeba_kid)

@register_metric
def moco_vit_church_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_vit_church'
    moco_vit_church_kid = kernel_inception_distance.compute_kid(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_vit_church_kid50k_full=moco_vit_church_kid)

@register_metric
def moco_vit_afhq_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_vit_afhq'
    moco_vit_afhq_kid = kernel_inception_distance.compute_kid(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_vit_afhq_kid50k_full=moco_vit_afhq_kid)

@register_metric
def moco_vit_lsunhorse_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_vit_lsunhorse'
    moco_vit_lsunhorse_kid = kernel_inception_distance.compute_kid(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_vit_lsunhorse_kid50k_full=moco_vit_lsunhorse_kid)

@register_metric
def moco_vit_lsunbedroom_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_vit_lsunbedroom'
    moco_vit_lsunbedroom_kid = kernel_inception_distance.compute_kid(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_vit_lsunbedroom_kid50k_full=moco_vit_lsunbedroom_kid)

@register_metric
def moco_vit_s0_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_vit_s0'
    moco_vit_s0_kid = kernel_inception_distance.compute_kid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(moco_vit_s0_kid50k_full=moco_vit_s0_kid)

@register_metric
def moco_vit_s1_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_vit_s1'
    moco_vit_s1_kid = kernel_inception_distance.compute_kid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(moco_vit_s1_kid50k_full=moco_vit_s1_kid)

@register_metric
def moco_vit_s2_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_vit_s2'
    mocov2_vit_s2_kid = kernel_inception_distance.compute_kid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(moco_vit_s2_kid50k_full=mocov2_vit_s2_kid)

#----------------------------------------------------------------------------
# Legacy metrics.

@register_metric
def fid50k(opts):
    opts.dataset_kwargs.update(max_size=None)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    fid = frechet_inception_distance.compute_fid(opts,  )
    return dict(fid50k=fid)

@register_metric
def kid50k(opts):
    opts.dataset_kwargs.update(max_size=None)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    kid = kernel_inception_distance.compute_kid(opts)
    return dict(kid50k=kid)

@register_metric
def pr50k3(opts):
    opts.dataset_kwargs.update(max_size=None)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    # opts.feature_network = 'swav'
    PR = precision_recall.compute_pr(opts,  nhood_size=3, row_batch_size=1000, col_batch_size=1000, detector_url=None)
    print(PR)
    return dict(pr50k3=PR)

@register_metric
def is50k(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    mean, std = inception_score.compute_is(opts, num_splits=10)
    return dict(is50k_mean=mean, is50k_std=std)

#----------------------------------------------------------------------------
# CKA.
@register_metric
def cka_full_numpy(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    cka = center_kernel_alignment_numpy.compute_cka(opts)
    return dict(cka_full_numpy=cka)

@register_metric
def cka_full_torch(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    cka = center_kernel_alignment_torch.compute_cka(opts)
    return dict(cka_full_torch=cka)

@register_metric
def swavcka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'swav'
    swavcka = center_kernel_alignment_torch.compute_cka(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(swavcka50k_full=swavcka)

@register_metric
def clipcka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'clip'
    clipcka = center_kernel_alignment_torch.compute_cka(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(clipcka50k_full=clipcka)

@register_metric
def sprcka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'spr'
    sprcka = center_kernel_alignment_torch.compute_cka(opts, detector_url=detector_urls[opts.feature_network])
    return dict(sprcka50k_full=sprcka)

@register_metric
def fasterrcnn_r50_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'fasterrcnn_r50'
    fasterrcnn_r50_cka = center_kernel_alignment_torch.compute_cka(opts,  num_subsets=100, max_subset_size=1000)
    return dict(fasterrcnn_r50_cka50k_full=fasterrcnn_r50_cka)

@register_metric
def maskrcnn_r50_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'maskrcnn_r50'
    maskrcnn_r50_cka = center_kernel_alignment_torch.compute_cka(opts,  num_subsets=100, max_subset_size=1000)
    return dict(maskrcnn_r50_cka50k_full=maskrcnn_r50_cka)

@register_metric
def vggcka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'vgg'
    vggcka = center_kernel_alignment_torch.compute_cka(opts,  num_subsets=100, max_subset_size=1000)
    return dict(vggcka50k_full=vggcka)

@register_metric
def moco_v2_r50_i_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_v2_r50_i'
    moco_v2_r50_i_cka = center_kernel_alignment_torch.compute_cka(opts,  num_subsets=100, max_subset_size=5000,detector_url=detector_urls[opts.feature_network])
    return dict(moco_v2_r50_i_cka50k_full=moco_v2_r50_i_cka)

@register_metric
def moco_r50_i_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_r50_i'
    moco_r50_i_cka = center_kernel_alignment_torch.compute_cka(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_r50_i_cka50k_full=moco_r50_i_cka)

@register_metric
def moco_r50_f_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_r50_f'
    moco_r50_f_cka = center_kernel_alignment_torch.compute_cka(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_r50_f_cka50k_full=moco_r50_f_cka)

@register_metric
def moco_r50_anime_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_r50_anime'
    moco_r50_anime_cka = center_kernel_alignment_torch.compute_cka(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(moco_r50_anime_cka50k_full=moco_r50_anime_cka)

@register_metric
def moco_r50_celeba_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_r50_celeba'
    moco_r50_celeba_cka = center_kernel_alignment_torch.compute_cka(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_r50_celeba_cka50k_full=moco_r50_celeba_cka)

@register_metric
def moco_r50_church_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_r50_church'
    moco_r50_church_cka = center_kernel_alignment_torch.compute_cka(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_r50_church_cka50k_full=moco_r50_church_cka)

@register_metric
def moco_r50_afhq_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_r50_afhq'
    moco_r50_afhq_cka = center_kernel_alignment_torch.compute_cka(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_r50_afhq_cka50k_full=moco_r50_afhq_cka)

@register_metric
def moco_r50_lsunhorse_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_r50_lsunhorse'
    moco_r50_lsunhorse_cka = center_kernel_alignment_torch.compute_cka(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_r50_lsunhorse_cka50k_full=moco_r50_lsunhorse_cka)

@register_metric
def moco_r50_lsunbedroom_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_r50_lsunbedroom'
    moco_r50_lsunbedroom_cka = center_kernel_alignment_torch.compute_cka(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_r50_lsunbedroom_cka50k_full=moco_r50_lsunbedroom_cka)

@register_metric
def moco_r50_s0_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_r50_s0'
    moco_r50_s0_cka = center_kernel_alignment_torch.compute_cka(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_r50_s0_cka50k_full=moco_r50_s0_cka)

@register_metric
def moco_r50_s1_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_r50_s1'
    moco_r50_s1_cka = center_kernel_alignment_torch.compute_cka(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_r50_s1_cka50k_full=moco_r50_s1_cka)

@register_metric
def moco_r50_s2_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_r50_s2'
    mocov2_r50_s2_cka = center_kernel_alignment_torch.compute_cka(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_r50_s2_cka50k_full=mocov2_r50_s2_cka)

@register_metric
def moco_vit_i_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_vit_i'
    moco_vit_i_cka = center_kernel_alignment_torch.compute_cka(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_vit_i_cka50k_full=moco_vit_i_cka)

@register_metric
def moco_vit_f_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_vit_f'
    moco_vit_f_cka = center_kernel_alignment_torch.compute_cka(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_vit_f_cka50k_full=moco_vit_f_cka)

@register_metric
def moco_vit_anime_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_vit_anime'
    moco_vit_anime_cka = center_kernel_alignment_torch.compute_cka(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(moco_vit_anime_cka50k_full=moco_vit_anime_cka)

@register_metric
def moco_vit_celeba_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_vit_celeba'
    moco_vit_celeba_cka = center_kernel_alignment_torch.compute_cka(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_vit_celeba_cka50k_full=moco_vit_celeba_cka)

@register_metric
def moco_vit_church_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_vit_church'
    moco_vit_church_cka = center_kernel_alignment_torch.compute_cka(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_vit_church_cka50k_full=moco_vit_church_cka)

@register_metric
def moco_vit_afhq_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_vit_afhq'
    moco_vit_afhq_cka = center_kernel_alignment_torch.compute_cka(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_vit_afhq_cka50k_full=moco_vit_afhq_cka)

@register_metric
def moco_vit_lsunhorse_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_vit_lsunhorse'
    moco_vit_lsunhorse_cka = center_kernel_alignment_torch.compute_cka(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_vit_lsunhorse_cka50k_full=moco_vit_lsunhorse_cka)

@register_metric
def moco_vit_lsunbedroom_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_vit_lsunbedroom'
    moco_vit_lsunbedroom_cka = center_kernel_alignment_torch.compute_cka(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_vit_lsunbedroom_cka50k_full=moco_vit_lsunbedroom_cka)

@register_metric
def moco_vit_s0_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_vit_s0'
    moco_vit_s0_cka = center_kernel_alignment_torch.compute_cka(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_vit_s0_cka50k_full=moco_vit_s0_cka)

@register_metric
def moco_vit_s1_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_vit_s1'
    moco_vit_s1_cka = center_kernel_alignment_torch.compute_cka(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_vit_s1_cka50k_full=moco_vit_s1_cka)

@register_metric
def moco_vit_s2_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'moco_vit_s2'
    mocov2_vit_s2_cka = center_kernel_alignment_torch.compute_cka(opts, detector_url=detector_urls[opts.feature_network])
    return dict(moco_vit_s2_cka50k_full=mocov2_vit_s2_cka)

# CLIP VIT-B-16
@register_metric
def clip_vit_B16_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'clip_vit_B16'
    clip_vit_B16_cka = center_kernel_alignment_torch.compute_cka(opts, detector_url=detector_urls[opts.feature_network])
    return dict(clip_vit_B16_cka50k_full=clip_vit_B16_cka)

@register_metric
def clip_vit_B16_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'clip_vit_B16'
    clip_vit_B16_kid = kernel_inception_distance.compute_kid(opts, detector_url=detector_urls[opts.feature_network])
    return dict(clip_vit_B16_kid50k_full=clip_vit_B16_kid)

@register_metric
def clip_vit_B16_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'clip_vit_B16'
    clip_vit_B16_fid = frechet_inception_distance.compute_fid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(clip_vit_B16_fid50k_full=clip_vit_B16_fid)

# clip VIT-B-32
@register_metric
def clip_vit_B32_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'clip_vit_B32'
    clip_vit_B32_cka = center_kernel_alignment_torch.compute_cka(opts, detector_url=detector_urls[opts.feature_network])
    return dict(clip_vit_B32_cka50k_full=clip_vit_B32_cka)

@register_metric
def clip_vit_B32_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'clip_vit_B32'
    clip_vit_B32_kid = kernel_inception_distance.compute_kid(opts, detector_url=detector_urls[opts.feature_network])
    return dict(clip_vit_B32_kid50k_full=clip_vit_B32_kid)

@register_metric
def clip_vit_B32_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'clip_vit_B32'
    clip_vit_B32_fid = frechet_inception_distance.compute_fid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(clip_vit_B32_fid50k_full=clip_vit_B32_fid)

# clip VIT-L-14
@register_metric
def clip_vit_L14_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'clip_vit_L14'
    clip_vit_L14_cka = center_kernel_alignment_torch.compute_cka(opts, detector_url=detector_urls[opts.feature_network])
    return dict(clip_vit_L14_cka50k_full=clip_vit_L14_cka)

@register_metric
def clip_vit_L14_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'clip_vit_L14'
    clip_vit_L14_kid = kernel_inception_distance.compute_kid(opts, detector_url=detector_urls[opts.feature_network])
    return dict(clip_vit_L14_kid50k_full=clip_vit_L14_kid)

@register_metric
def clip_vit_L14_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'clip_vit_L14'
    clip_vit_L14_fid = frechet_inception_distance.compute_fid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(clip_vit_L14_fid50k_full=clip_vit_L14_fid)

@register_metric
def convnext_base_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'convnext_base'
    convnext_base_fid = frechet_inception_distance.compute_fid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(convnext_base_fid50k_full=convnext_base_fid)

@register_metric
def convnext_base_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'convnext_base'
    convnext_base_kid = kernel_inception_distance.compute_kid(opts,  detector_url=detector_urls[opts.feature_network])
    return dict(convnext_base_kid50k_full=convnext_base_kid)

@register_metric
def convnext_base_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'convnext_base'
    convnext_base_cka = center_kernel_alignment_torch.compute_cka(opts, detector_url=detector_urls[opts.feature_network])
    return dict(convnext_base_cka50k_full=convnext_base_cka)

@register_metric
def vitcls_base_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'vitcls_base_patch16_224'
    vitcls_base_fid = frechet_inception_distance.compute_fid(opts)
    return dict(vitcls_base_fid50k_full=vitcls_base_fid)

@register_metric
def vitcls_base_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'vitcls_base_patch16_224'
    vitcls_base_kid = kernel_inception_distance.compute_kid(opts)
    return dict(vitcls_base_kid50k_full=vitcls_base_kid)

@register_metric
def vitcls_base_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'vitcls_base_patch16_224'
    vitcls_base_cka = center_kernel_alignment_torch.compute_cka(opts)
    return dict(vitcls_base_cka50k_full=vitcls_base_cka)

@register_metric
def swin_base_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'swin_base_patch4_window7_224'
    swin_base_fid = frechet_inception_distance.compute_fid(opts)
    return dict(swin_base_fid50k_full=swin_base_fid)

@register_metric
def swin_base_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'swin_base_patch4_window7_224'
    swin_base_kid = kernel_inception_distance.compute_kid(opts)
    return dict(swin_base_kid50k_full=swin_base_kid)

@register_metric
def swin_base_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'swin_base_patch4_window7_224'
    swin_base_cka = center_kernel_alignment_torch.compute_cka(opts)
    return dict(swin_base_cka50k_full=swin_base_cka)

@register_metric
def deit_base_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'deit_base_patch16_224'
    deit_base_fid = frechet_inception_distance.compute_fid(opts)
    return dict(deit_base_fid50k_full=deit_base_fid)

@register_metric
def deit_base_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'deit_base_patch16_224'
    deit_base_kid = kernel_inception_distance.compute_kid(opts)
    return dict(deit_base_kid50k_full=deit_base_kid)

@register_metric
def deit_base_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'deit_base_patch16_224'
    deit_base_cka = center_kernel_alignment_torch.compute_cka(opts)
    return dict(deit_base_cka50k_full=deit_base_cka)

@register_metric
def repvgg_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'repvgg_b3'
    repvgg_fid = frechet_inception_distance.compute_fid(opts)
    return dict(repvgg_fid50k_full=repvgg_fid)

@register_metric
def repvgg_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'repvgg_b3'
    repvgg_kid = kernel_inception_distance.compute_kid(opts)
    return dict(repvgg_kid50k_full=repvgg_kid)

@register_metric
def repvgg_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'repvgg_b3'
    repvgg_cka = center_kernel_alignment_torch.compute_cka(opts)
    return dict(repvgg_cka50k_full=repvgg_cka)

@register_metric
def resmlp_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'resmlp_24_224_dino'
    resmlp_fid = frechet_inception_distance.compute_fid(opts)
    return dict(resmlp_fid50k_full=resmlp_fid)

@register_metric
def resmlp_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'resmlp_24_224_dino'
    resmlp_kid = kernel_inception_distance.compute_kid(opts)
    return dict(resmlp_kid50k_full=resmlp_kid)

@register_metric
def resmlp_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'resmlp_24_224_dino'
    resmlp_cka = center_kernel_alignment_torch.compute_cka(opts)
    return dict(resmlp_cka50k_full=resmlp_cka)

@register_metric
def mixermlp_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'mixer_b16_224'
    mixermlp_fid = frechet_inception_distance.compute_fid(opts)
    return dict(mixermlp_fid50k_full=mixermlp_fid)

@register_metric
def mixermlp_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'mixer_b16_224'
    mixermlp_kid = kernel_inception_distance.compute_kid(opts)
    return dict(mixermlp_kid50k_full=mixermlp_kid)

@register_metric
def mixermlp_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'mixer_b16_224'
    mixermlp_cka = center_kernel_alignment_torch.compute_cka(opts)
    return dict(mixermlp_cka50k_full=mixermlp_cka)

@register_metric
def gmlp_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'gmlp_s16_224'
    gmlp_fid = frechet_inception_distance.compute_fid(opts)
    return dict(gmlp_fid50k_full=gmlp_fid)

@register_metric
def gmlp_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'gmlp_s16_224'
    gmlp_kid = kernel_inception_distance.compute_kid(opts)
    return dict(gmlp_kid50k_full=gmlp_kid)

@register_metric
def gmlp_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'gmlp_s16_224'
    gmlp_cka = center_kernel_alignment_torch.compute_cka(opts)
    return dict(gmlp_cka50k_full=gmlp_cka)

@register_metric
def maxxvitv2_rmlp_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'maxxvitv2_rmlp_base_rw_224'
    maxxvitv2_rmlp_fid = frechet_inception_distance.compute_fid(opts)
    return dict(maxxvitv2_rmlp_fid50k_full=maxxvitv2_rmlp_fid)

@register_metric
def maxxvitv2_rmlp_kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'maxxvitv2_rmlp_base_rw_224'
    maxxvitv2_rmlp_kid = kernel_inception_distance.compute_kid(opts)
    return dict(maxxvitv2_rmlp_kid50k_full=maxxvitv2_rmlp_kid)

@register_metric
def maxxvitv2_rmlp_cka50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.generate_dataset_kwargs.update(max_size=None, xflip=False)
    opts.feature_network = 'maxxvitv2_rmlp_base_rw_224'
    maxxvitv2_rmlp_cka = center_kernel_alignment_torch.compute_cka(opts)
    return dict(maxxvitv2_rmlp_cka50k_full=maxxvitv2_rmlp_cka)

@register_metric
def f_inception_h_inception_cka10k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    feature_network_f=None
    feature_network_h=None
    f_inception_h_inception_cka = center_kernel_alignment_revelance.compute_cka(opts,feature_network_f=None,feature_network_h=None,detector_url_f=None,detector_url_h=None)
    return dict(f_inception_h_inception_cka10k_full=f_inception_h_inception_cka)

@register_metric
def f_inception_h_convnext_cka10k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    feature_network_f=None
    feature_network_h='convnext_base'
    f_inception_h_convnext_cka = center_kernel_alignment_revelance.compute_cka(opts,feature_network_f=None,feature_network_h='convnext_base',detector_url_f=None,detector_url_h=detector_urls[feature_network_h])
    return dict(f_inception_h_convnext_cka10k_full=f_inception_h_convnext_cka)

@register_metric
def f_inception_h_vitcls_cka10k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    feature_network_f=None
    feature_network_h='vitcls_base_patch16_224'
    f_inception_h_vitcls_cka = center_kernel_alignment_revelance.compute_cka(opts,feature_network_f=None,feature_network_h='vitcls_base_patch16_224',detector_url_f=None,detector_url_h=None)
    return dict(f_inception_h_vitcls_cka10k_full=f_inception_h_vitcls_cka)

@register_metric
def f_convnext_h_inception_cka10k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    feature_network_f='convnext_base'
    feature_network_h=None
    f_convnext_h_inception_cka = center_kernel_alignment_revelance.compute_cka(opts,feature_network_f='convnext_base',feature_network_h=None,detector_url_f=detector_urls[feature_network_f],detector_url_h=None)
    return dict(f_convnext_h_inception_cka10k_full=f_convnext_h_inception_cka)

@register_metric
def f_convnext_h_convnext_cka10k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    feature_network_f='convnext_base'
    feature_network_h='convnext_base'
    f_convnext_h_convnext_cka = center_kernel_alignment_revelance.compute_cka(opts,feature_network_f='convnext_base',feature_network_h='convnext_base',detector_url_f=detector_urls[feature_network_f],detector_url_h=detector_urls[feature_network_h])
    return dict(f_convnext_h_convnext_cka10k_full=f_convnext_h_convnext_cka)

@register_metric
def f_convnext_h_vitcls_cka10k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    feature_network_f='convnext_base'
    feature_network_h='vitcls_base_patch16_224'
    f_convnext_h_vitcls_cka = center_kernel_alignment_revelance.compute_cka(opts,feature_network_f='convnext_base',feature_network_h='vitcls_base_patch16_224',detector_url_f=detector_urls[feature_network_f],detector_url_h=None)
    return dict(f_convnext_h_vitcls_cka10k_full=f_convnext_h_vitcls_cka)

@register_metric
def f_vitcls_h_inception_cka10k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    feature_network_f='vitcls_base_patch16_224'
    feature_network_h=None
    f_vitcls_h_inception_cka = center_kernel_alignment_revelance.compute_cka(opts,feature_network_f='vitcls_base_patch16_224',feature_network_h=None,detector_url_f=None,detector_url_h=None)
    return dict(f_vitcls_h_inception_cka10k_full=f_vitcls_h_inception_cka)

@register_metric
def f_vitcls_h_convnext_cka10k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    feature_network_f='vitcls_base_patch16_224'
    feature_network_h='convnext_base'
    f_vitcls_h_convnext_cka = center_kernel_alignment_revelance.compute_cka(opts,feature_network_f='vitcls_base_patch16_224',feature_network_h='convnext_base',detector_url_f=None,detector_url_h=detector_urls[feature_network_h])
    return dict(f_vitcls_h_convnext_cka10k_full=f_vitcls_h_convnext_cka)

@register_metric
def f_vitcls_h_vitcls_cka10k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    feature_network_f='vitcls_base_patch16_224'
    feature_network_h='vitcls_base_patch16_224'
    f_vitcls_h_vitcls_cka = center_kernel_alignment_revelance.compute_cka(opts,feature_network_f='vitcls_base_patch16_224',feature_network_h='vitcls_base_patch16_224',detector_url_f=None,detector_url_h=None)
    return dict(f_vitcls_h_vitcls_cka10k_full=f_vitcls_h_vitcls_cka)

@register_metric
def f_moco_vit_h_inception_cka10k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    feature_network_f='moco_vit_i'
    feature_network_h=None
    f_moco_vit_h_inception_cka = center_kernel_alignment_revelance.compute_cka(opts,feature_network_f='moco_vit_i',feature_network_h=None,detector_url_f=detector_urls[feature_network_f],detector_url_h=None)
    return dict(f_moco_vit_h_inception_cka10k_full=f_moco_vit_h_inception_cka)

@register_metric
def f_moco_vit_h_convnext_cka10k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    feature_network_f='moco_vit_i'
    feature_network_h='convnext_base'
    f_moco_vit_h_convnext_cka = center_kernel_alignment_revelance.compute_cka(opts,feature_network_f='moco_vit_i',feature_network_h='convnext_base',detector_url_f=detector_urls[feature_network_f],detector_url_h=detector_urls[feature_network_h])
    return dict(f_moco_vit_h_convnext_cka10k_full=f_moco_vit_h_convnext_cka)

@register_metric
def f_moco_vit_h_vitcls_cka10k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    feature_network_f='moco_vit_i'
    feature_network_h='vitcls_base_patch16_224'
    f_moco_vit_h_vitcls_cka = center_kernel_alignment_revelance.compute_cka(opts,feature_network_f='moco_vit_i',feature_network_h='vitcls_base_patch16_224',detector_url_f=detector_urls[feature_network_f],detector_url_h=None)
    return dict(f_moco_vit_h_vitcls_cka10k_full=f_moco_vit_h_vitcls_cka)

@register_metric
def f_clip_vit_h_inception_cka10k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    feature_network_f='clip_vit_B32'
    feature_network_h=None
    f_clip_vit_h_inception_cka = center_kernel_alignment_revelance.compute_cka(opts,feature_network_f='clip_vit_B32',feature_network_h=None,detector_url_f=detector_urls[feature_network_f],detector_url_h=None)
    return dict(f_clip_vit_h_inception_cka10k_full=f_clip_vit_h_inception_cka)

@register_metric
def f_clip_vit_h_convnext_cka10k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    feature_network_f='clip_vit_B32'
    feature_network_h='convnext_base'
    f_clip_vit_h_convnext_cka = center_kernel_alignment_revelance.compute_cka(opts,feature_network_f='clip_vit_B32',feature_network_h='convnext_base',detector_url_f=detector_urls[feature_network_f],detector_url_h=detector_urls[feature_network_h])
    return dict(f_clip_vit_h_convnext_cka10k_full=f_clip_vit_h_convnext_cka)

@register_metric
def f_clip_vit_h_vitcls_cka10k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    feature_network_f='clip_vit_B32'
    feature_network_h='vitcls_base_patch16_224'
    f_clip_vit_h_vitcls_cka = center_kernel_alignment_revelance.compute_cka(opts,feature_network_f='clip_vit_B32',feature_network_h='vitcls_base_patch16_224',detector_url_f=detector_urls[feature_network_f],detector_url_h=None)
    return dict(f_clip_vit_h_vitcls_cka10k_full=f_clip_vit_h_vitcls_cka)

@register_metric
def cka_similarity_detectors(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    cka_similarity = center_kernel_alignment_revelance.compute_cka(opts)
    return dict(cka_similarity_detectors=cka_similarity)
