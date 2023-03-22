'''
Some functions we use
'''

import numpy as np
import torch
import transformers
import random
import os
from sklearn.linear_model import LinearRegression as LinReg

import utils.pickle as pck

#############################


def verbose(x):
    with open('log.txt', 'a') as f:
        f.write(x + '\n')


# create missing folders on the way, just a convenience...
def file(*args):
    lst = [arg for arg in args]
    if isinstance(lst[0], list):
        lst = lst[0]
    result = ''
    assert len(lst) >= 2
    if len(lst) > 2:
        folder_to_create = ''
        for folder in lst[:-2]:
            folder_to_create += folder
            try:
                os.mkdir(folder_to_create)
            except FileExistsError:
                pass
            folder_to_create += '/'
    result += ''.join([fld + '/' for fld in lst[:-2]])
    result += lst[-2] + '.' + lst[-1]
    return result


def rng(start, end, steps, no_start=False, no_end=False):
    s = 1 if no_start else 0
    e = steps if no_end else steps+1

    return [start + i*(end-start)/steps for i in range(s, e)]


# get model from transformers by name
def get_model(model_name):
    if 'gpt2' in model_name:
        model =\
            transformers.GPT2Model.from_pretrained(model_name,
                                                   output_hidden_states=True)
        model.eval()
        return model
    elif 'bert' in model_name:
        model =\
            transformers.BertModel.from_pretrained(model_name,
                                                   output_hidden_states=True)
        model.eval()
        return model
    else:
        raise RuntimeError('model not supported...')


# get tokenizer from transformers by name
def get_tokenizer(model_name):
    if 'gpt2' in model_name:
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.add_special_tokens({'pad_token': '.'})
        return tokenizer
    elif 'bert' in model_name:
        tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
        return tokenizer
    else:
        raise RuntimeError('model not supported...')


def mul(A, v):
    if len(A.shape) != 2:
        raise ValueError('A must be of dimension 2')
    if len(v.shape) == 0:
        raise ValueError('v must be of dimension > 0')
    if A.shape[-1] != v.shape[-1]:
        raise ValueError('the last dimension of A should coincide with '
                         'the last dimension of v')
    return (A @ v[..., None]).squeeze(-1)


def randomize_below(v):
    if isinstance(v, np.ndarray):
        w = v.copy()
    elif isinstance(v, torch.Tensor):
        w = v.clone()
    else:
        raise TypeError(f'v is not allowed to be of type {type(v)}.')
    for i in range(w.shape[0]):
        w[i] = random.randrange(int(w[i]))
    return w


def concat(v, w):
    if v is None:
        return w
    elif isinstance(v, np.ndarray) and isinstance(w, np.ndarray):
        return np.concatenate((v, w), axis=0)
    elif isinstance(v, torch.Tensor) and isinstance(w, torch.Tensor):
        return torch.cat((v, w), dim=0)
    elif isinstance(v, dict) and isinstance(w, dict):
        return {key: concat(v[key], w[key]) for key in v}
    else:
        return v + w


def concat_iter(iterable):
    result = None
    for a in iterable:
        result = concat(result, a)
    return result


# used only for the BERT experiment
def load_vectors(model_folder_name, file_name):
    vectors = {}
    for data_bit in pck.load_all('experiment/' + model_folder_name + '/' +
                                 file_name + '.pickle'):
        for layer in data_bit:
            if layer not in vectors:
                vectors[layer] = data_bit[layer]
            else:
                vectors[layer] = concat(vectors[layer],
                                        data_bit[layer])
    vectors['num_of_layers'] = len([layer for layer
                                    in vectors
                                    if isinstance(layer, int)]) - 1
    vectors['num_of_samples'] = vectors[0].shape[0]
    return vectors


def linreg(x, y, intercept=False,
           file_name=None, keep_device=False):
    if keep_device:
        device = x.device

    x = x.detach().cpu()
    y = y.detach().cpu()

    reg = LinReg(fit_intercept=intercept).fit(x, y)
    if intercept:
        reg = [torch.from_numpy(reg.coef_),
               torch.from_numpy(reg.intercept_)]
    else:
        reg = torch.from_numpy(reg.coef_)

    if file_name is not None:
        pck.save(reg, file_name)

    if keep_device:
        if intercept:
            reg = [c.to(device) for c in reg]
        else:
            reg = reg.to(device)

    return reg


def load_mat(model_folder_name, indices, dataset=None, device=None):
    file_name_list = ['linreg', model_folder_name]
    if dataset is not None:
        file_name_list += [dataset]
    if len(indices) == 2:
        file_name_list += ['_'.join([str(indices[0]),
                                     str(indices[1])])]
    elif len(indices) == 3:
        file_name_list += ['_'.join(['p' + str(indices[0]),
                                     str(indices[1]),
                                     str(indices[2])])]
    else:
        raise RuntimeError()
    file_name_list += ['pickle']
    file_name = file(file_name_list)
    A = pck.load(file_name)
    if device is not None:
        if isinstance(A, list):
            A = [c.to(device) for c in A]
        else:
            A = A.to(device)
    return A


# this does not use batched functions
# so maybe there is a more elegant implementation...
def jaccard_sim(v, w, k_v, k_w=None):
    if k_w is None:
        k_w = k_v
    assert len(v.shape) == 2 and len(w.shape) == 2,\
        'v and w should have shapes of length 2'
    assert v.shape[0] == w.shape[0],\
        'v.shape[0] must be equal to w.shape[0]'
    assert v.shape[-1] >= k_v and w.shape[-1] >= k_w,\
        'k_v or k_w are too large'
    n = v.shape[0]
    jac_sum = 0.
    for i in range(n):
        intersection = float(torch.isin(v[i][:k_v], w[i][:k_w]).sum())
        union = k_v + k_w - intersection
        jac = intersection / union
        jac_sum += jac
    return jac_sum / n


# this does not use batched functions
# so maybe there is a more elegant implementation...
def is_in(v, w, k_w=10):
    assert len(v.shape) == 1 and len(w.shape) == 2,\
        'v should have shape of length 1 and w should have shape of length 2'
    assert v.shape[0] == w.shape[0],\
        'v.shape[0] must be equal to w.shape[0]'
    assert w.shape[-1] >= k_w,\
        'k_w is too large'
    n = v.shape[0]
    isin_sum = 0.
    for i in range(n):
        if torch.isin(v[i], w[i, :k_w]):
            isin_sum += 1.
    return isin_sum / n


def is_equal2(v, w, u):
    assert len(v.shape) == 1 and len(w.shape) == 1 and len(u.shape) == 1,\
        'v,w and u should have shape of length 1'
    assert v.shape[0] == w.shape[0] and w.shape[0] == u.shape[0],\
        'v.shape[0] must be equal to w.shape[0] and to u.shape[0]'
    v_cooc = (v == u).float()
    w_cooc = (w == u).float()
    vw_cooc = v_cooc * w_cooc
    v_cooc = v_cooc.mean().item()
    w_cooc = w_cooc.mean().item()
    vw_cooc = vw_cooc.mean().item()
    return v_cooc, w_cooc, vw_cooc
