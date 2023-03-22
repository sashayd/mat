import torch

#############################


def get_cuda_if_possible(verbose=False):
    if verbose == True:
        verbose = print

    devices = [d for d in range(torch.cuda.device_count())]
    device_names = [torch.cuda.get_device_name(d) for d in devices]
    if verbose:
        verbose('CUDA devices: ', device_names)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        if verbose:
            verbose('Will use GPU for training.')
    else:
        device = torch.device('cpu')
        if verbose:
            verbose('Will use CPU for training.')
    return device


def select_1_by_0(v, indices):
    sh = list(v.shape)
    d = sh[0]
    n = len(sh)
    sh[1] = 1
    sh1 = [1]*n
    sh1[0] = d
    indices = indices.view(sh1)
    indices = indices.expand(sh)
    return v.gather(dim=1, index=indices).squeeze(1)


def select_2_by_01(v, indices):
    sh = list(v.shape)
    d = sh[0]
    e = sh[1]
    n = len(sh)
    sh[2] = 1
    sh1 = [1]*n
    sh1[0] = d
    sh1[1] = e
    indices = indices.view(sh1)
    indices = indices.expand(sh)
    return v.gather(dim=2, index=indices).squeeze(2)
