# GAN_Metrics
Evaluation system of image generation using GAN.

## Release notes

This repository is based on [stylegan3](https://github.com/NVlabs/stylegan3), with several new abilities:
- Tools for the heat map visualization of the influence of the feature distribution of different feature extractors on the evaluation results ( `grad_cam.py` ).
- Tools for matching the histogram of the number of labels for generated dataset with the histogram of real dataset ( `label_match.py` , `gen_match.py` , `histogram.py` ).
- Centered Kernel Alignment metrics ( `cka` ).
- Multi-level for metrics ( `layers` ).
- General improvements: a new and stable evaluation system of image generation using GAN.

Compatibility:
- Compatible with old metrics and operations of [stylegan3](https://github.com/NVlabs/stylegan3).

## Requirements

* Linux and Windows are supported, but we recommend Linux for performance and compatibility reasons.
* 1&ndash;8 high-end NVIDIA GPUs with at least 12 GB of memory. We have done all testing and development using Tesla V100 and A100 GPUs.
* 64-bit Python 3.8 and PyTorch 1.9.0 (or later). See https://pytorch.org for PyTorch install instructions.
* CUDA toolkit 11.1 or later.  (Why is a separate CUDA toolkit installation required?  See [Troubleshooting](./docs/troubleshooting.md#why-is-cuda-toolkit-installation-necessary)).
* GCC 7 or later (Linux) or Visual Studio (Windows) compilers.  Recommended GCC version depends on CUDA version, see for example [CUDA 11.4 system requirements](https://docs.nvidia.com/cuda/archive/11.4.1/cuda-installation-guide-linux/index.html#system-requirements).
* Python libraries: see [environment.yml](./environment.yml) for exact library dependencies.  You can use the following commands with Miniconda3 to create and activate your StyleGAN3 Python environment:
  - `conda env create -f environment.yml`
  - `conda activate stylegan3`
* Docker users:
  - Ensure you have correctly installed the [NVIDIA container runtime](https://docs.docker.com/config/containers/resource_constraints/#gpu).
  - Use the [provided Dockerfile](./Dockerfile) to build an image with the required library dependencies.

The code relies heavily on custom PyTorch extensions that are compiled on the fly using NVCC. On Windows, the compilation requires Microsoft Visual Studio. We recommend installing [Visual Studio Community Edition](https://visualstudio.microsoft.com/vs/) and adding it into `PATH` using `"C:\Program Files (x86)\Microsoft Visual Studio\<VERSION>\Community\VC\Auxiliary\Build\vcvars64.bat"`.

See [Troubleshooting](./docs/troubleshooting.md) for help on common installation and run-time problems.

## Getting started

See [stylegan3](https://github.com/NVlabs/stylegan3) for basic operations such as generating images and training...

### Quality metrics

```.bash
# Pre-trained network pickle: specify dataset explicitly, print result to stdout and save result to txt.
python calc_metrics.py \
        --metrics=fid50k_full \
        --data /mnt/petrelfs/zhangyichi/data/ffhq256_50k.zip \
        --eval_bs=1000 \
        --layers=Conv2d_4a_3x3,Mixed_5d,Mixed_6e,Mixed_7c \
        --mirror=1 \
        --cache=1 \
        --feature_save_flag=1 \
        --cfg=stylegan2 \
        --random=0 \
        --max_real=50000 \
        --num_gen=50000 \
        --save_name=ffhq_full_vs_ffhq_50K_random_new_set \
        --generate /mnt/petrelfs/zhangyichi/generate_datasets/ffhq/random_50K_ffhq \
        --network /mnt/petrelfs/zhangyichi/fid/stylegan2-ffhq-256x256.pkl \
```

### Grad-cam

```.bash
# Metrics:fid/kid/cka, choose detector for grad_cam and save result images to html.
python -u grad_cam.py \
        --metrics=fid \
        --detectors=inception,clip,moco_vit_i,clip_vit_B16 \
        --stats_path /mnt/petrelfs/zhangyichi/stats/mu_sigma \
        --html_name=visualize_fid_ffhq \
        --generate_image_path /mnt/petrelfs/zhangyichi/generate_datasets/ffhq_cam \
        --outdir /mnt/petrelfs/zhangyichi/grad_cam/processed_ffhq_fid \
```

### Label-statistics

```.bash
# The real dataset label statistics using inception_v3/resnet50.
python -u label_match.py \
        --real_dataset /mnt/petrelfs/zhangyichi/data/ffhq256_50k.zip \
        --inception_label /mnt/petrelfs/zhangyichi/data/real_ffhq_labels_inception.pickle \
        --resnet50_label /mnt/petrelfs/zhangyichi/data/real_ffhq_labels_resnet50.pickle \
```

### Generate-match

```.bash
# Generate images matching real dataset, save images to outdir.
python -u gen_match.py \
        --seeds=8000000-8100000 \
        --trunc=1 \
        --limit=0.001 \
        --cfg=stylegan2 \
        --num_real=50000 \
        --inception_label /mnt/petrelfs/zhangyichi/data/real_ffhq_labels_inception.pickle \
        --detector=inception \
        --outdir /mnt/petrelfs/zhangyichi/generate_datasets/ffhq/match_inception_ffhq_final \
        --network /mnt/petrelfs/zhangyichi/fid/stylegan2-ffhq-256x256.pkl \
```

### Histogram

```.bash
# Choose which real dataset and generate dataset for histogram with which detector. 
python -u histogram.py \
        --real_dataset /mnt/petrelfs/zhangyichi/data/ffhq256_50k.zip \
        --gen_dataset /mnt/petrelfs/zhangyichi/generate_datasets/ffhq/match_inception_ffhq_new \
        --detector inception_v3 \
        --histogram_save /mnt/petrelfs/zhangyichi/histogram/ffhq_inceptionset_inception.png \
```


Recommended metrics:
* `moco_vit_i_cka50k_full`: Centered Kernel Alignment using MOCO-ViT exactor trained on ImageNet dataset.
* `fid50k_full`: Fr&eacute;chet inception distance<sup>[1]</sup> against the full dataset.
* `kid50k_full`: Kernel inception distance<sup>[2]</sup> against the full dataset.
* `pr50k3_full`: Precision and recall<sup>[3]</sup> againt the full dataset.
* `ppl2_wend`: Perceptual path length<sup>[4]</sup> in W, endpoints, full image.
* `eqt50k_int`: Equivariance<sup>[5]</sup> w.r.t. integer translation (EQ-T).
* `eqt50k_frac`: Equivariance w.r.t. fractional translation (EQ-T<sub>frac</sub>).
* `eqr50k`: Equivariance w.r.t. rotation (EQ-R).

Legacy metrics:
* `cka50k`: Centered Kernel Alignment against 50k real images.
* `fid50k`: Fr&eacute;chet inception distance against 50k real images.
* `kid50k`: Kernel inception distance against 50k real images.
* `pr50k3`: Precision and recall against 50k real images.
* `is50k`: Inception score<sup>[6]</sup> for CIFAR-10.

References:
1. [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500), Heusel et al. 2017
2. [Demystifying MMD GANs](https://arxiv.org/abs/1801.01401), Bi&nacute;kowski et al. 2018
3. [Improved Precision and Recall Metric for Assessing Generative Models](https://arxiv.org/abs/1904.06991), Kynk&auml;&auml;nniemi et al. 2019
4. [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948), Karras et al. 2018
5. [Alias-Free Generative Adversarial Networks](https://nvlabs.github.io/stylegan3), Karras et al. 2021
6. [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498), Salimans et al. 2016

