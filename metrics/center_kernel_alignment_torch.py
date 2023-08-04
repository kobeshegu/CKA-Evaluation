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

def softmax3d(input, softmax_mode):
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


def seed_everything(seed):
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)

def fusion_features(features, opts, group, last):
    assert opts.fusion_ways in ['sum', 'cat'], f'fusion_ways must be `sum` or `cat`!'
    # group_index = group_index.split(',')
    length = len(features)


    # sum operation
    if opts.fusion_ways == 'sum':
        if group == 0:
            if opts.fusion_softmax_order == 'pre_fusion':
                fused_feature = softmax3d(features[0], softmax_mode=opts.cka_normalize)
                for index in range(1, int(float(opts.group_index[0]))):
                    fused_feature += softmax3d(features[index+1], softmax_mode=opts.cka_normalize)
            else:
                fused_feature = features[0]
                for index in range(1, int(float(opts.group_index[0]))):
                    fused_feature += features[index+1]
        else:
            if last:
                if opts.fusion_softmax_order == 'pre_fusion':
                    fused_feature = softmax3d(features[int(float(opts.group_index[group-1]))+1], softmax_mode=opts.cka_normalize)
                    for index in range(int(float(opts.group_index[group-1])) + 2, length):
                        fused_feature += softmax3d(features[index], softmax_mode=opts.cka_normalize)
                else:   
                    fused_feature = features[int(float(opts.group_index[group-1]))+1]
                    for index in range(int(float(opts.group_index[group-1])) + 2, length):
                        fused_feature += features[index]   
            else:
                if opts.fusion_softmax_order == 'pre_fusion':
                    fused_feature = softmax3d(features[int(float(opts.group_index[group-1]))+1], softmax_mode=opts.cka_normalize)
                    for index in range(int(float(opts.group_index[group-1])) + 2, int(float(opts.group_index[group])) + 1):
                        fused_feature += softmax3d(features[index], softmax_mode=opts.cka_normalize)
                else:
                    fused_feature = features[int(float(opts.group_index[group-1]))+1]
                    for index in range(int(float(opts.group_index[group-1])) + 2, int(float(opts.group_index[group])) + 1):
                        fused_feature += features[index]
    else:
        if group == 0:
            if opts.fusion_softmax_order == 'pre_fusion':
                fused_feature = softmax3d(features[0], softmax_mode=opts.cka_normalize)
                for index in range(1, int(float(opts.group_index[0]))):
                    fused_feature = torch.cat((fused_feature, softmax3d(features[index+1],softmax_mode=opts.cka_normalize)), dim=1)
            else:
                fused_feature = features[0]
                for index in range(1, int(float(opts.group_index[0]))):
                    fused_feature = torch.cat((fused_feature, features[index+1]), dim=1)
        else:
            if last:
                if opts.fusion_softmax_order == 'pre_fusion':
                    fused_feature = softmax3d(features[int(float(opts.group_index[group-1]))+1], softmax_mode=opts.cka_normalize)
                    for index in range(int(float(opts.group_index[group-1])) + 2, length):
                        fused_feature = torch.cat((fused_feature, softmax3d(features[index],softmax_mode=opts.cka_normalize)), dim=1)
                else:
                    fused_feature = features[int(float(opts.group_index[group-1]))+1]
                    for index in range(int(float(opts.group_index[group-1])) + 2, length):
                        fused_feature = torch.cat((fused_feature, features[index]), dim=1)   
            else:
                if opts.fusion_softmax_order == 'pre_fusion':
                    fused_feature = softmax3d(features[int(float(opts.group_index[group-1]))+1], softmax_mode=opts.cka_normalize)
                    for index in range(int(float(opts.group_index[group-1])) + 2, int(float(opts.group_index[group])) + 1):
                        fused_feature = torch.cat((fused_feature, softmax3d(features[index],softmax_mode=opts.cka_normalize)), dim=1)
                else:
                    fused_feature = features[int(float(opts.group_index[group-1]))+1]
                    for index in range(int(float(opts.group_index[group-1])) + 2, int(float(opts.group_index[group])) + 1):
                        fused_feature = torch.cat((fused_feature, features[index]), dim=1)
    return fused_feature

def random_projection(features, out_channels):
    in_channels = features.shape[1]
    out_channels = out_channels
    seed_everything(0)
    fc_layer = torch.nn.Linear(in_channels, out_channels).to('cuda')
    torch.nn.init.normal_(fc_layer.weight, mean=0.0, std=0.01)
    return_features = fc_layer(features.to('cuda'))
    return return_features

def fake_calculate(opts, f, res_real, detector_url,detector_kwargs):
    cka_res={}
    num_gen=opts.num_gen
    #real_dataset
    feature_real={}

    #generation
    feature_gen={}
    f.write("--------new-epoch---------\n")
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
        cka_res[layer], _ , _, _, _ =cka_cal(res_real[layer].get_all(), res_gen[layer].get_all(), opts)
        rew=model+'_'+layer+':'+str(cka_res[layer])+'\n'
        f.write(rew)
    return cka_res


def cka_cal(real_features, gen_features, opts):
    if opts.rank != 0:
        return float('nan')
    # print(f'real_features:', real_features.shape)
    # print(f'gen_features:', gen_features.shape)

    real_features = torch.tensor(real_features)
    gen_features = torch.tensor(gen_features)
    if opts.cka_normalize is not None and opts.fusion_softmax_order == 'post_fusion':
        assert opts.cka_normalize in ['3d', '2d', 'L1', 'L2'], f'cka_normalize dimension must be one of `3d`, `2d`, `L1` and `L2` !'
        real_features = softmax3d(real_features, softmax_mode=opts.cka_normalize)
        gen_features = softmax3d(gen_features, softmax_mode=opts.cka_normalize)
    m = min(min(real_features.shape[0], gen_features.shape[0]), opts.max_subset_size)
    cka = 0
    cka_minors = 0
    assert opts.dimension in ['N', 'C'], f'CKA dimension must be `N` or `C`!'
    if opts.dimension == 'N':
        print(f'calculating cka of dimension n*n for {opts.num_subsets} times with {opts.max_subset_size} subset size')
    else:
        print(f'calculating cka of dimension c*c for {opts.num_subsets} times with {opts.max_subset_size} subset size')

    # begin calculate CKA
    if opts.subset_estimate:
        for _subset_idx in range(opts.num_subsets):
            one_time = time.time()
            x = gen_features[np.random.choice(gen_features.shape[0], m, replace=False)]
            y = real_features[np.random.choice(real_features.shape[0], m, replace=False)]

            # features (N,C,H,W), flatten
            # if not opts.post_process:
            #     x = torch.flatten(x, start_dim=1)
            #     y = torch.flatten(y, start_dim=1)
            # dim 
            if opts.dimension =='N':
                L_real_features = torch.matmul(y, y.T)
                L_gen_features = torch.matmul(x, x.T)
            else:
                L_real_features = torch.matmul(y.T, y)
                L_gen_features = torch.matmul(x.T, x)
            # kernel type 
            if opts.kernel == 'rbf':
                L_real_features = rbf_kernel(L_real_features)
                L_gen_features = rbf_kernel(L_gen_features)
            elif opts.kernel == 'poly':
                L_real_features = poly_kernel(L_real_features, m=m)
                L_gen_features = poly_kernel(L_gen_features, m=m)
            centering_real_features = centering(L_real_features)  # KH
            centering_gen_features = centering(L_gen_features)  # LH
            hsic = torch.sum(centering_real_features * centering_gen_features)  # trace property: sum of element-wise multiplication = trace(matrix multiplication)
            var1 = torch.sqrt(torch.sum(centering_real_features * centering_real_features))  
            var2 = torch.sqrt(torch.sum(centering_gen_features * centering_gen_features))
            cka += hsic / (var1 * var2)
            cka_minors = hsic - (var1 * var2)
            one_time_time = time.time() - one_time
            print(f'CKA (of time {_subset_idx}) is {((hsic / (var1 * var2))).numpy()}, Total CKA is {cka.numpy()}, Calculation time:{one_time_time} s.')
        # print(hsic, var1, var2, cka)
        cka = cka / opts.num_subsets 
        cka_minors = cka_minors / opts.num_subsets
        return float(cka.numpy()), float(var1.numpy()), float(var2.numpy()), float(hsic.numpy()), float(cka_minors.numpy())
    else:
        x = gen_features
        y = real_features
        if opts.dimension =='N':
            L_real_features = torch.matmul(y, y.T)
            L_gen_features = torch.matmul(x, x.T)
        else:
            L_real_features = torch.matmul(y.T, y)
            L_gen_features = torch.matmul(x.T, x)
        # kernel type 
        if opts.kernel == 'rbf':
            L_real_features = rbf_kernel(L_real_features)
            L_gen_features = rbf_kernel(L_gen_features)
        elif opts.kernel == 'poly':
            L_real_features = poly_kernel(L_real_features, m=m)
            L_gen_features = poly_kernel(L_gen_features, m=m)
        centering_real_features = centering(L_real_features)  # KH
        centering_gen_features = centering(L_gen_features)  # LH
        hsic = torch.sum(centering_real_features * centering_gen_features)  # trace property: sum of element-wise multiplication = trace(matrix multiplication)
        var1 = torch.sqrt(torch.sum(centering_real_features * centering_real_features))  
        var2 = torch.sqrt(torch.sum(centering_gen_features * centering_gen_features))
        cka = hsic / (var1 * var2)
        cka_minors = hsic - (var1 * var2)
        return float(cka.numpy()), float(var1.numpy()), float(var2.numpy()), float(hsic.numpy()), float(cka_minors.numpy())

def compute_cka(opts, detector_url=None):
    if detector_url is None:
        detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True)  # Return raw features before the softmax layer.
    if opts.feature_network is None:
        model='inception'
    elif opts.feature_network == 'spr':
        model='resnet50'
    else:
        model=opts.feature_network
    if opts.rank != 0:
        return float('nan')
    os.makedirs(opts.save_res, exist_ok=True)
    output_name = opts.save_res  + '/' + 'cka_'+opts.save_name+'.txt'
    f = open(output_name,'a')
    f.write("-----------new-metrics------------\n")

    #real_dataset
    res_real = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, batch_size=opts.eval_bs, capture_all=True, capture_mean_cov=True, max_items=opts.max_real, layers=opts.layers)

    #generate_dataset & calculate
    if opts.random == True:
        cka_list=[]
        cka_mean={}
        cka_std={}
        f1_name=opts.save_res +'/' + 'cka_mean_'+opts.save_name+'.txt'
        f2_name=opts.save_res +'/' + 'cka_std_'+opts.save_name+'.txt'
        f1=open(f1_name,'a')
        f1.write("--------new-epoch----------\n")
        f2=open(f2_name,'a')
        f2.write("--------new-epoch----------\n")
        for num in range(opts.random_num):
            print('Epoch: %d' %num)
            cka_list.append(fake_calculate(opts, f, res_real, detector_url,detector_kwargs))
        for layer in opts.layers:
            res=[]
            for num in range(opts.random_num):
                res.append(cka_list[num][layer])
            res=np.array(res)
            cka_mean[layer]=np.mean(res)
            cka_std[layer]=np.std(res)
            f1_res=model+'_'+layer+':'+str(cka_mean[layer])+'\n'
            f1.write(f1_res)
            f2_res=model+'_'+layer+':'+str(cka_std[layer])+'\n'
            f2.write(f2_res)
        f1.close()
        f2.close()
        f.close()
        return cka_mean

    else:
        cka_res={}
        cka_minors={}
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
            feature_real[layer]=res_real[layer].get_all_torch()
            feature_gen[layer]=res_gen[layer].get_all_torch()
            cka_res[layer], var1, var2, hsic, cka_minors[layer]=cka_cal(feature_real[layer], feature_gen[layer], opts)
            rew=model+'_'+layer+'_'+opts.dimension+':'+str(cka_res[layer])+'\n'
            f.write(rew)
            rew_minor = model+'_cka_minor'+layer+'_'+opts.dimension+':'+str(cka_minors[layer])+'\n'
            #f.write(rew_minor)
            if opts.save_real_features:
                dir=f'{opts.feature_save}'
                model_path=os.path.join(dir,model)
                path=os.path.join(model_path,layer)
                os.makedirs(path, exist_ok=True)
                filename1 = path+'/feature_real.pickle'
                with open(filename1,'wb') as fo1:
                    pickle.dump(feature_real[layer],fo1,protocol = pickle.HIGHEST_PROTOCOL)
                    fo1.close()
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
            # hsic_x = var1 / (feature_gen[layer].shape[0])**2
            # hsic_y = var2 / (feature_real[layer].shape[0])**2
            # hsic_xy = np.sqrt(hsic) / (feature_real[layer].shape[0])**2
            #trklhl = 'tr(xhxh):'+ str(var1) + '\n' + 'tr(yhyh):' + str(var2) +'\n' + 'tr(xhyh):' + str(hsic) + '\n'
            #f.write(trklhl)
            # hsic = 'hsic(x,x):'+ str(hsic_x) + '\n' + 'hsic(y,y):' + str(hsic_y) +'\n' + 'hsic(x,y):' + str(hsic_xy) + '\n'
            # f.write(f'{hsic}\n')

        # combine multi-level features
        feature_real_list = list(feature_real.values())
        feature_gen_list = list(feature_gen.values())
        feature_real_group = {}
        feature_gen_group = {}
        cka_res_fusion = {}
        if opts.groups > 0:
            for group in range(opts.groups):
                if group == opts.groups - 1:
                    feature_real_group[group] = fusion_features(feature_real_list, opts, group, last=True)
                    feature_gen_group[group]  = fusion_features(feature_gen_list, opts, group, last=True)
                else:
                    feature_real_group[group] = fusion_features(feature_real_list, opts, group, last=False)
                    feature_gen_group[group]  = fusion_features(feature_gen_list, opts, group, last=False)
                print(feature_real_group[group].shape)
                if opts.random_projection:
                    feature_gen_group[group] = random_projection(feature_gen_group[group], out_channels=768).detach().cpu().numpy()
                    feature_real_group[group] = random_projection(feature_real_group[group], out_channels=768).detach().cpu().numpy()
                    print(feature_real_group[group].shape)
                cka_res_fusion[group], _, _, _, _ = cka_cal(feature_real_group[group], feature_gen_group[group], opts)
                print(cka_res_fusion[group])
                multi_res = model+'fused_group_'+str(group)+':'+str(cka_res_fusion[group])+'\n'
                f.write(multi_res)
        if opts.fuse_all:
            if opts.fusion_ways == 'sum':
                feature_real_fuse_all = feature_real_group[0]
                feature_gen_fuse_all = feature_gen_group[0]
                for i in range(1, len(feature_gen_group)):
                    feature_real_fuse_all = torch.cat((feature_real_fuse_all, feature_real_group[i]), dim=1)
                    feature_gen_fuse_all  = torch.cat((feature_gen_fuse_all, feature_gen_group[i]), dim=1)
                cka_res_fuse_all, _, _, _, _ = cka_cal(feature_real_fuse_all, feature_gen_fuse_all, opts)
                print(cka_res_fuse_all)
                fuse_all_res = model+'fused_all'+opts.fusion_ways+'_:'+str(cka_res_fuse_all)+'\n'
                f.write(fuse_all_res)
                f.close()
            else:
                for i in range(0, len(feature_gen_group)):
                    feature_real_group[i] = random_projection(feature_real_group[i], out_channels=768).detach().cpu()
                    feature_gen_group[i]  = random_projection(feature_gen_group[i], out_channels=768).detach().cpu()
                fuse_all_feature_real = feature_real_group[0]
                fuse_all_feature_gen  = feature_gen_group[0]
                if opts.fuse_all_ways == 'cat':
                    for i in range(1, len(feature_gen_group)):
                        fuse_all_feature_real = torch.cat((fuse_all_feature_real, feature_real_group[i]), dim=1)
                        fuse_all_feature_gen  = torch.cat((fuse_all_feature_gen, feature_gen_group[i]), dim=1)
                else:
                    for i in range(1, len(feature_gen_group)):
                        fuse_all_feature_real += feature_real_group[i]
                        fuse_all_feature_gen  += feature_gen_group[i]
                cka_res_fuse_all, _, _, _, _ = cka_cal(fuse_all_feature_real, fuse_all_feature_gen, opts)
                print(cka_res_fuse_all)
                fuse_all_res = model+opts.fusion_ways+'fused_all_'+opts.fuse_all_ways+':'+str(cka_res_fuse_all)+'\n'
                f.write(fuse_all_res)
                f.close()                
        return cka_res
