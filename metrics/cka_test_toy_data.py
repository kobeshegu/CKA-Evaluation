mport math
import numpy as np
import torch
import random
import scipy.linalg
from cleanfid import fid # pip install clean-fid first

def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    #return np.dot(np.dot(H, K), H)  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    return np.dot(K, H)  # KH

def centering_torch(K):
    n = K.shape[0]
    unit = torch.ones([n, n], device=K.device)
    I = torch.eye(n, device=K.device)
    H = I - unit / n
    #return torch.matmul(torch.matmul(H, K), H)
    return torch.matmul(K, H)

def rbf_kernel(GX, sigma=None):
    GX = torch.matmul(GX, GX.T)
    KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
    if sigma is None:
        mdist = torch.median(KX[KX!=0])
        sigma = math.sqrt(mdist)
    KX *= -0.5 / (sigma * sigma)
    KX = torch.exp(KX)
    return KX

def rbf(X, sigma=None):
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = np.exp(KX)
    return KX

def cka_cal(real_features, gen_features, kernel=False):
    real_features = torch.tensor(real_features)
    gen_features = torch.tensor(gen_features)
    x = gen_features
    y = real_features
    L_real_features = torch.matmul(x.T, x)
    print(L_real_features.shape)
    L_gen_features = torch.matmul(y.T, y)
    if kernel == True:
        centering_real_features = centering_torch(rbf_kernel(L_real_features).to(torch.float32))  # KH
        centering_gen_features = centering_torch(rbf_kernel(L_gen_features).to(torch.float32))  # LH
    else:
        centering_real_features = centering_torch(L_real_features.to(torch.float32))  # KH
        centering_gen_features = centering_torch(L_gen_features.to(torch.float32))  # LH
    hsic = torch.sum(centering_real_features * centering_gen_features)  # trace property: sum of element-wise multiplication = trace(matrix multiplication)
    var1 = torch.sqrt(torch.sum(centering_real_features * centering_real_features))
    var2 = torch.sqrt(torch.sum(centering_gen_features * centering_gen_features))
    cka = hsic / (var1 * var2)
    return float(cka)

def kernel_HSIC(X, Y, sigma):
    L_X = np.dot(X.T, X)
    L_Y = np.dot(Y.T, Y)
    return np.sum(centering(rbf(L_X, sigma)) * centering(rbf(L_Y, sigma)))

def linear_HSIC(X, Y):
    L_X = np.dot(X.T, X)
    L_Y = np.dot(Y.T, Y)
    # print(L_X.shape)
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

def seed_everything(seed):
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)


def kid_cal(real_features, gen_features):
    n = real_features.shape[1]
    # print(n)
    x = gen_features
    y = real_features
    a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
    b = (x @ y.T / n + 1) ** 3
    kid = (a.sum() - np.diag(a).sum()) / n - b.sum() * 2 /n
    return float(kid/n)

# ---------------------------------------------------------------------------


if __name__=='__main__':
    seed = 0
    seed_everything(seed)
    X = np.random.randn(1000, 100)
    print(X.mean())
    print(X.std())
    Y = np.random.randn(1000, 100)
    print(Y.mean())
    print(Y.std())
    # transpose: X * X.T
    X = X.transpose(1, 0)
    Y = Y.transpose(1, 0)
    print('---------------Numpy_CKA----------------')
    print('Linear CKA, between X and Y: {}'.format(linear_CKA(X, Y)))
    print('Linear CKA, between X and X: {}'.format(linear_CKA(X, X)))
    #
    print('RBF Kernel CKA, between X and Y: {}'.format(kernel_CKA(X, Y)))
    print('RBF Kernel CKA, between X and X: {}'.format(kernel_CKA(X, X)))
    print('---------------Torch_CKA----------------')
    print('Linear CKA, between X and Y: {}'.format(cka_cal(X, Y)))
    print('Linear CKA, between X and X: {}'.format(cka_cal(X, X)))
    #
    print('RBF Kernel CKA, between X and Y: {}'.format(cka_cal(X, Y,kernel=True)))
    print('RBF Kernel CKA, between X and X: {}'.format(cka_cal(X, X, kernel=True)))

    print('---------------KID_results----------------')
    print('KID, between X and Y: {}'.format(kid_cal(X, Y)))
    print('KID, between X and X: {}'.format(kid_cal(X, X)))
    print('KID, between Y and Y: {}'.format(kid_cal(Y, Y)))

    print('---------------Clean-fid:KID_results----------------')
    print('Clean-fid-KID, between X and Y: {}'.format(fid.kernel_distance(X, Y)))
    print('Clean-fid-KID, between X and X: {}'.format(fid.kernel_distance(X, X)))
    print('Clean-fid-KID, between Y and Y: {}'.format(fid.kernel_distance(Y, Y)))
