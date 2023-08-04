import torch
import time
from training.networks_stylegan3 import grid_sample_crop

t0 = time.time()
x = torch.ones([8, 2, 256, 256]).cuda()
t1 = time.time()
out = grid_sample_crop(x, 1, 256, 256)
t2 = time.time()
print(t2-t1, t1-t0)


