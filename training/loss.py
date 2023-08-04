# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from training.networks_stylegan3 import differetiable_crop, grid_sample_crop
from torch_utils.ops import grid_sample_gradfix

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------
grad_list  = list()

def grad_hook(module, grad_input, grad_output):
    if grad_output[0] is None:
        #print(module, 'grad is none')
        return None
    #print(grad_output[0].shape)
    grad_list.append(grad_output[0].detach())

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0, crop_flag=False, crop_ratio=1.0, crop_when_real=False, crop_when_fake=False, scale_range=None, offset_per_instance=False, full_scale_prob=0, lw_rec=0, capture_grads=True, batch_sampler=1):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.crop_flag          = crop_flag
        self.crop_ratio         = crop_ratio
        self.crop_when_real     = crop_when_real
        self.crop_when_fake     = crop_when_fake
        self.scale_range        = scale_range
        self.offset_per_instance = offset_per_instance
        self.full_scale_prob = full_scale_prob
        self.lw_rec = lw_rec
        self.capture_grads = capture_grads
        self.grad_keys = list()
        self.batch_sampler = batch_sampler
        if capture_grads:
            try:
                self.G.mapping.fc0.register_full_backward_hook(grad_hook)
                self.G.mapping.fc1.register_full_backward_hook(grad_hook)
                self.G.synthesis.input.affine.register_full_backward_hook(grad_hook)
                self.grad_keys.insert(0, 'G_mapping0')
                self.grad_keys.insert(0, 'G_mapping1')
                self.grad_keys.insert(0, 'G_input_affine')

                if hasattr(self.D, 'b128.fromrgb'):
                    self.D.b128.fromrgb.register_full_backward_hook(grad_hook)
                    self.grad_keys.insert(0, 'D_b128_fromrgb')
                if hasattr(self.D, 'b4.conv'):
                    self.D.b4.conv.register_full_backward_hook(grad_hook)
                    self.grad_keys.insert(0, 'D_b4_conv')
                if hasattr(self.D, 'b4.fc'):
                    self.D.b4.fc.register_full_backward_hook(grad_hook)
                    self.grad_keys.insert(0, 'D_b4_fc')
                if hasattr(self.D, 'b4.out'):
                    self.D.b4.out.register_full_backward_hook(grad_hook)
                    self.grad_keys.insert(0, 'D_b4_out')
            except:
                self.G.module.mapping.fc0.register_full_backward_hook(grad_hook)
                self.G.module.mapping.fc1.register_full_backward_hook(grad_hook)
                self.G.module.synthesis.input.affine.register_full_backward_hook(grad_hook)
                self.grad_keys.insert(0, 'G_mapping0')
                self.grad_keys.insert(0, 'G_mapping1')
                self.grad_keys.insert(0, 'G_input_affine')

                if hasattr(self.D.module, 'b128.fromrgb'):
                    self.D.module.b128.fromrgb.register_full_backward_hook(grad_hook)
                    self.grad_keys.insert(0, 'D_b128_fromrgb')
                if hasattr(self.D.module, 'b4.conv'):
                    self.D.module.b4.conv.register_full_backward_hook(grad_hook)
                    self.grad_keys.insert(0, 'D_b4_conv')
                if hasattr(self.D.module, 'b4.fc'):
                    self.D.module.b4.fc.register_full_backward_hook(grad_hook)
                    self.grad_keys.insert(0, 'D_b4_fc')
                if hasattr(self.D.module, 'b4.out'):
                    self.D.module.b4.out.register_full_backward_hook(grad_hook)
                    self.grad_keys.insert(0, 'D_b4_out')


    def run_G(self, z, c, update_emas=False):
        ws = self.G.mapping(z, c, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        img, sampling_grid = self.G.synthesis(ws, update_emas=update_emas, crop_flag=self.crop_flag and not self.crop_when_fake)
        return img, ws, sampling_grid

    def run_G_full(self, z, c, update_emas=False):
        ws = self.G.mapping(z, c, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        img, sampling_grid = self.G.synthesis(ws, update_emas=update_emas) 
        return img, ws, sampling_grid

    def run_D(self, img, c, blur_sigma=0, update_emas=False, real_input=True, log_mbstd=None):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        # crop
        do_crop = self.crop_when_real if real_input else self.crop_when_fake
        if do_crop and self.crop_flag:
            img, sampling_grid = grid_sample_crop(img, self.crop_ratio, int(img.shape[3]), int(img.shape[2]), scale_range=self.scale_range, offset_per_instance=self.offset_per_instance, full_scale_prob=self.full_scale_prob)
        logits = self.D(img, c, update_emas=update_emas, log_mbstd=log_mbstd)
        return logits

    def cal_rec_loss(self, gen_img_full, gen_img, sampling_grid):
        batch_size = gen_img_full.shape[0] 
        device = gen_img_full.device
        # Build unit grid
        w, h = gen_img_full.shape[3], gen_img_full.shape[2]
        unit_grid_x = torch.linspace(-1.0, 1.0, w)[None, None, :, None].repeat(batch_size, w, 1, 1)
        unit_grid_y = torch.linspace(-1.0, 1.0, h)[None, None, :, None].repeat(batch_size, h, 1, 1).transpose(1, 2)
        unit_grid = torch.cat([unit_grid_x, unit_grid_y], dim=3).to(device)
        sampling_grid_on_img = unit_grid * sampling_grid[0] + sampling_grid[1]
        gen_img_full_crop = grid_sample_gradfix.grid_sample(gen_img_full, sampling_grid_on_img)
        rec_loss = torch.nn.functional.mse_loss(gen_img_full_crop, gen_img.detach())
        return rec_loss

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.pl_weight == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        global grad_list

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                if self.batch_sampler > 1:
                    gen_z = gen_z[::self.batch_sampler]
                    gen_c = gen_c[::self.batch_sampler]
                gen_img, _gen_ws, sampling_grid = self.run_G(gen_z, gen_c)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, real_input=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)
                if self.lw_rec > 0:
                    gen_img_full, _, _ = self.run_G_full(gen_z, gen_c)
                    rec_loss = self.cal_rec_loss(gen_img_full, gen_img, sampling_grid)
                    loss_Gmain = loss_Gmain + rec_loss * self.lw_rec

            with torch.autograd.profiler.record_function('Gmain_backward'):
                grad_list = list()
                loss_Gmain.mean().mul(gain).backward()

            if phase == 'Gmain' and self.capture_grads:
                for i, op_name in enumerate(self.grad_keys):
                    training_stats.report(f'Grads_Gtraining_linf/{op_name}', grad_list[i].max())
                    training_stats.report(f'Grads_Gtraining_l2/{op_name}', grad_list[i].square().mean())
                

        # Gpl: Apply path length regularization.
        if phase in ['Greg', 'Gboth']:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws, sampling_grid = self.run_G(gen_z[:batch_size], gen_c[:batch_size])
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients(self.pl_no_weight_grad):
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                loss_Gpl.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws, sampling_grid = self.run_G(gen_z, gen_c, update_emas=True)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True, real_input=False, log_mbstd='fake')
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma, real_input=True, log_mbstd='real' if name == 'Dreal' else None)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
