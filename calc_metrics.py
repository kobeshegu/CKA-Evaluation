# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Calculate quality metrics for previous training run or pretrained network pickle."""

import os
import string
import click
import json
import tempfile
import copy
import torch

import dnnlib
import legacy
from metrics import metric_main
from metrics import metric_utils
from torch_utils import training_stats
from torch_utils import custom_ops
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

#----------------------------------------------------------------------------
#torch.backends.cudnn.enabled = False

def subprocess_fn(rank, args, temp_dir):
    dnnlib.util.Logger(should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0 or not args.verbose:
        custom_ops.verbosity = 'none'

    # Configure torch.
    device = torch.device('cuda', rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    conv2d_gradfix.enabled = True

    if args.network_pkl is not None:
        # Print network summary.
        G = copy.deepcopy(args.G).eval().requires_grad_(False).to(device)
        if rank == 0 and args.verbose:
            z = torch.empty([1, G.z_dim], device=device)
            c = torch.empty([1, G.c_dim], device=device)
            # misc.print_module_summary(G, [z, c])
    else:
        G = None

    # Calculate each metric.
    for metric in args.metrics:
        if rank == 0 and args.verbose:
            print(f'Calculating {metric}...')
        progress = metric_utils.ProgressMonitor(verbose=args.verbose)
        result_dict = metric_main.calc_metric(metric=metric, G=G, dataset_kwargs=args.dataset_kwargs, generate_dataset_kwargs=args.generate_dataset_kwargs, 
        num_gpus=args.num_gpus,generate=args.generate, rank=rank, device=device, progress=progress, eval_bs=args.eval_bs, 
        resolution=args.resolution, cache=args.cache, feature_save_flag=args.feature_save_flag, layers=args.layers, random=args.random, random_size=args.random_size, 
        random_num=args.random_num, cfg=args.cfg, dimension=args.dimension, kernel=args.kernel, sigma=args.sigma, save_res=args.save_res, 
        feature_save=args.feature_save, save_stats=args.save_stats, num_gen=args.num_gen, save_name=args.save_name, max_real=args.max_real, 
        token_vit=args.token_vit, cka_normalize=args.cka_normalize, post_process=args.post_process, subset_estimate=args.subset_estimate, 
        num_subsets=args.num_subsets, max_subset_size=args.max_subset_size, groups=args.groups, group_index=args.group_index, fusion_ways=args.fusion_ways, 
        fusion_softmax_order=args.fusion_softmax_order, random_projection=args.random_projection, fuse_all=args.fuse_all, fuse_all_ways=args.fuse_all_ways,
        detector1=args.detector1, detector2=args.detector2, save_real_features=args.save_real_features)
        if rank == 0:
            metric_main.report_metric(result_dict, run_dir=args.run_dir, snapshot_pkl=args.network_pkl)
        if rank == 0 and args.verbose:
            print()

    # Done.
    if rank == 0 and args.verbose:
        print('Exiting...')

#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('network_pkl', '--network', help='Network pickle filename or URL', default= None, metavar='PATH', show_default=True)
@click.option('--metrics', help='Quality metrics', metavar='[NAME|A,B,C|none]', type=parse_comma_separated_list, default='mixermlp_fid50k_full', show_default=True)
@click.option('--data', help='Dataset to evaluate against  [default: look up]', default="D:/Z-kobeshegu/NeurIPS2023-rebuttal/random_50K_ffhq.zip", metavar='[ZIP|DIR]', show_default=True)
@click.option('--generate', help='Generated dataset to evaluate', default="D:/Z-kobeshegu/NeurIPS2023-rebuttal/random_50K_ffhq.zip", metavar='[ZIP|DIR]', show_default=True)
@click.option('--mirror', help='Enable dataset x-flips  [default: look up]', type=bool, default=1, metavar='BOOL', show_default=True)
@click.option('--gpus', help='Number of GPUs to use', type=int, default=1, metavar='INT', show_default=True)
@click.option('--eval_bs', help='batch_size', type=int, default=1, metavar='INT', show_default=True)
@click.option('--resolution', help='resolution', type=int, default=256, metavar='INT', show_default=True)
@click.option('--cache', help='whether to look up from cache', type=bool, default=0, metavar='BOOL', show_default=True)
@click.option('--feature_save_flag', help='whether to save features', type=bool, default=0, metavar='BOOL', show_default=True)
@click.option('--layers', help='Which layer to get features', type=parse_comma_separated_list, default='blocks.2', metavar='STRING', show_default=True)
@click.option('--random', help='whether to use random for generate dataset', type=bool, default=False, metavar='BOOL', show_default=True)
@click.option('--random_size', help='random_size', type=int, default=5000, metavar='INT', show_default=True)
@click.option('--random_num', help='random_num', type=int, default=5, metavar='INT', show_default=True)
@click.option('--cfg', help='Base configuration', type=click.Choice(['stylegan3', 'stylegan2']), default='stylegan2')
@click.option('--verbose', help='Print optional information', type=bool, default=True, metavar='BOOL', show_default=True)
# for cka computation
@click.option('--dimension', help='Compute CKA for N*N or C*C [default: C*C]', type=str, default='C', metavar='BOOL', show_default=True)
@click.option('--kernel', help='What kernel to use for computing cka', type=str, default='rbf', metavar='STRING', show_default=True)
@click.option('--sigma', help='Kernel parameter of RBF kernel, valid for rbf kernel only', type=float, default=None, metavar='FLOAT', show_default=True)
@click.option('--save_res', help='Where to save the results of computed metrics', type=str, default='D:/Z-kobeshegu/NeurIPS2023-rebuttal/results', metavar='STRING', show_default=True)
@click.option('--feature_save', help='Where to save the features of different dataset', type=str, default='D:/Z-kobeshegu/NeurIPS2023-rebuttal/features', metavar='STRING', show_default=True)
@click.option('--save_stats', help='Where to save the stats of computed metrics', type=str, default=None, metavar='STRING', show_default=True)
@click.option('--num_gen', help='Number of images to generate for computation when generate dataset is none', type=int, default=1000, metavar='INT', show_default=True)
@click.option('--save_name', help='Which file to save the results of computed metrics', type=str, default='test', metavar='STRING', show_default=True)
@click.option('--max_real', help='The max number of realdataset for computation', type=int, default=1000, metavar='INT', show_default=True)
@click.option('--token_vit', help='Take out the token feature of VIT or not', type=bool, default=True, metavar='BOOL', show_default=True)
@click.option('--cka_normalize', help='Whether to perform softmax on the features after the avgpool for CKA', type=str, default='2d', metavar='STRING', show_default=True)
# save forward featurs 
@click.option('--post_process', help='Whether to perform the avgpool on the features', type=str, default='mean', metavar='STRING', show_default=True)
@click.option('--subset_estimate', help='Estimating GT by choosing subset for multiple times', type=bool, default=True, metavar='BOOL', show_default=True)
# num_subsets & max_subset_size 
@click.option('--num_subsets', help='The number of times for estimating the GT of computing all samples', type=int, default=50, metavar='INT', show_default=True)
@click.option('--max_subset_size', help='The number of samples for each computation', type=int, default=5000, metavar='INT', show_default=True)
# multi-level features fusion
@click.option('--groups', help='How many groups combined in various layers', type=int, default=0, metavar='INT', show_default=True)
@click.option('--group_index', help='The layer index for grouping different layers', type=parse_comma_separated_list, default=None, metavar='STRING', show_default=True)
@click.option('--fusion_ways', help='How to combine multi-level features: concat or sum', type=str, default='sum', metavar='STRING', show_default=True)
@click.option('--fusion_softmax_order', help='The order of softmax for fusing features, `pre_fusion` or `post_fusion` ', type=str, default='post_fusion', metavar='STRING', show_default=True)
# random projection
@click.option('--random_projection', help='Perform random projection for more efficient computation', type=bool, default=False, metavar='BOOL', show_default=True)
@click.option('--fuse_all', help='Fuse low, mid, high-level features', type=bool, default=False, metavar='BOOL', show_default=True)
@click.option('--fuse_all_ways', help='How to fusing multi-level features: concat or sum', type=str, default=None, metavar='STRING', show_default=True)
# CKA similarity of feature extractors
@click.option('--detector1', help='Feature extractor 1 for computing cka similarity', type=str, default=None, metavar='STRING', show_default=True)
@click.option('--detector2', help='Feature extractor 2 for computing cka similarity', type=str, default=None, metavar='STRING', show_default=True)
@click.option('--save_real_features', help='save feature of real datasets for futher use', type=bool, default=None, metavar='BOOL', show_default=True)

def calc_metrics(ctx, network_pkl, metrics, data, generate,mirror, gpus, eval_bs, resolution, cache, feature_save_flag, layers, random, random_size, random_num, cfg, verbose, dimension, kernel, sigma, save_res, feature_save, save_stats, num_gen, save_name, max_real, token_vit, cka_normalize, post_process, subset_estimate, num_subsets, max_subset_size, groups, group_index, fusion_ways, fusion_softmax_order, random_projection, fuse_all, fuse_all_ways, detector1, detector2, save_real_features):
    """Calculate quality metrics for previous training run or pretrained network pickle.

    Examples:

    \b
    # Previous training run: look up options automatically, save result to JSONL file.
    python calc_metrics.py --metrics=eqt50k_int,eqr50k \\
        --network=~/training-runs/00000-stylegan3-r-mydataset/network-snapshot-000000.pkl

    \b
    # Pre-trained network pickle: specify dataset explicitly, print result to stdout.
    python calc_metrics.py --metrics=fid50k_full --data=~/datasets/ffhq-1024x1024.zip --mirror=1 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl

    \b
    Recommended metrics:
      fid50k_full  Frechet inception distance against the full dataset.
      kid50k_full  Kernel inception distance against the full dataset.
      pr50k3_full  Precision and recall againt the full dataset.
      ppl2_wend    Perceptual path length in W, endpoints, full image.
      eqt50k_int   Equivariance w.r.t. integer translation (EQ-T).
      eqt50k_frac  Equivariance w.r.t. fractional translation (EQ-T_frac).
      eqr50k       Equivariance w.r.t. rotation (EQ-R).

    \b
    Legacy metrics:
      fid50k       Frechet inception distance against 50k real images.
      kid50k       Kernel inception distance against 50k real images.
      pr50k3       Precision and recall against 50k real images.
      is50k        Inception score for CIFAR-10.
    """
    dnnlib.util.Logger(should_flush=True)

    # Validate arguments.
    args = dnnlib.EasyDict(metrics=metrics, num_gpus=gpus, network_pkl=network_pkl, verbose=verbose, eval_bs=eval_bs, 
    layers=layers, random=random, random_size=random_size, random_num=random_num, cfg=cfg, resolution=resolution, cache=cache, 
    feature_save_flag=feature_save_flag, generate=generate, dimension=dimension, kernel=kernel, sigma=sigma, save_res=save_res, 
    feature_save=feature_save, save_stats=save_stats, num_gen=num_gen, save_name=save_name, max_real=max_real, token_vit=token_vit, 
    cka_normalize=cka_normalize, post_process=post_process, subset_estimate=subset_estimate, num_subsets=num_subsets, max_subset_size=max_subset_size, 
    groups=groups, group_index=group_index, fusion_ways=fusion_ways, fusion_softmax_order=fusion_softmax_order, random_projection=random_projection, 
    fuse_all=fuse_all, fuse_all_ways=fuse_all_ways, detector1=detector1, detector2=detector2, save_real_features=save_real_features)
    if not all(metric_main.is_valid_metric(metric) for metric in args.metrics):
        ctx.fail('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    if not args.num_gpus >= 1:
        ctx.fail('--gpus must be at least 1')
    if network_pkl is not None:
        # Load network.
        if not dnnlib.util.is_url(network_pkl, allow_file_urls=True) and not os.path.isfile(network_pkl):
            ctx.fail('--network must point to a file or URL')
        if args.verbose:
            print(f'Loading network from "{network_pkl}"...')
        with dnnlib.util.open_url(network_pkl, verbose=args.verbose) as f:
            network_dict = legacy.load_network_pkl(cfg, f)
            args.G = network_dict['G_ema'] # subclass of torch.nn.Module

    # Initialize dataset options.
    if data is not None:
        args.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data)
    elif network_pkl is not None and network_dict['training_set_kwargs'] is not None:
        args.dataset_kwargs = dnnlib.EasyDict(network_dict['training_set_kwargs'])
    else:
        ctx.fail('Could not look up dataset options; please specify --data')

    args.generate_dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset',path=generate)
    
    # Finalize dataset options.
    if network_pkl is not None:
        args.dataset_kwargs.resolution = args.G.img_resolution
        args.dataset_kwargs.use_labels = (args.G.c_dim != 0)
        args.generate_dataset_kwargs.resolution = args.G.img_resolution
        args.generate_dataset_kwargs.use_labels = (args.G.c_dim != 0)
    else:
        args.dataset_kwargs.resolution = resolution
        args.dataset_kwargs.use_labels = False
        args.generate_dataset_kwargs.resolution = resolution
        args.generate_dataset_kwargs.use_labels = False
    if mirror is not None:
        args.dataset_kwargs.xflip = mirror
        args.generate_dataset_kwargs.xflip = mirror

    # Print dataset options.
    if args.verbose:
        print('Dataset options:')
        print(json.dumps(args.dataset_kwargs, indent=2))

    # Locate run dir.
    args.run_dir = None
    # if os.path.isfile(network_pkl):
    #     pkl_dir = os.path.dirname(network_pkl)
    #     if os.path.isfile(os.path.join(pkl_dir, 'training_options.json')):
    #         args.run_dir = pkl_dir

    # Launch processes.
    if args.verbose:
        print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    calc_metrics() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
