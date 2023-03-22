'''
Compute and save results for the paper (BERT)
TODO: add GPU option
'''

import torch
import transformers
import math

import utils.pickle as pck
import utils.torch as trc

from aux import mul, concat_iter, load_vectors, load_mat, rng, file, verbose

#############################

if __name__ != '__main__':
    raise RuntimeError('This script is not intended to be imported')

#############################

model_name = 'bert-base-uncased'
model_folder_name = model_name + '_mask'
dataset = 'wikipedia'
batch_size = 16
compute_jump = True
compute_early_exit = True

#############################

save_file_name = file('experiment', model_folder_name,
                      dataset + '_results', 'pickle')

vectors = load_vectors(model_folder_name, dataset + '_val')
num_of_layers = vectors['num_of_layers']
token_positions = vectors['token_positions']
tokenized_sentences = vectors['tokenized_sentences']

N = [token_positions[i]+1 for i in range(len(token_positions))]
N = sum(N) / len(N)
N = float(N)

maskedLMhead = transformers.BertForMaskedLM.from_pretrained(model_name).cls
maskedLMhead.eval()
tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

As = {}


def get_dist(vecs):
    vecs = vecs[:, None, :]
    with torch.no_grad():
        prediction_dist = maskedLMhead(vecs)[:, 0, :]
    prediction_dist = prediction_dist.softmax(dim=-1)
    return prediction_dist


final_layer_dist = get_dist(vectors[num_of_layers])


def early_exit_threshold(layer,
                         token_pos,
                         params,
                         plain=False):
    if plain:
        return params['lambda']
    if layer < params['minimal_layer']:
        return 2.
    else:
        result = 0.
        result += 0.9 * params['lambda']
        result += 0.1 * math.exp(
            -params['tau'] *
            token_pos /
            params['N']
            )
        return result


def early_exit(idx, mode, params):
    for layer in range(num_of_layers):
        vec = vectors[layer][idx][None, :]
        final_vec = vec
        if mode == 'mat' and layer != num_of_layers:
            if (layer, num_of_layers) not in As:
                As[(layer, num_of_layers)] =\
                    load_mat(model_folder_name, (layer, num_of_layers),
                             dataset=dataset)
            A = As[(layer, num_of_layers)]
            final_vec = mul(A, final_vec)
        dist = get_dist(final_vec)[0]
        top2 = dist.topk(k=2, dim=-1)[0]
        top2diff = (top2[0] - top2[1]).item()
        if top2diff > early_exit_threshold(layer, 3, params):
            topk = dist.topk(k=10, dim=-1)[1][None, :]
            return topk, layer
    final_vec = vectors[num_of_layers][idx][None, :]
    dist = get_dist(final_vec)[0]
    topk = dist.topk(k=10, dim=-1)[1][None, :]
    return topk, num_of_layers


config_list = []
for i in range(num_of_layers+1):
    config_list.append((i, 'id'))
for i in range(num_of_layers+1):
    config_list.append((i, 'mat'))

if compute_jump:
    for layer, mode in config_list:
        verbose(f'working on layer={layer} and mode={mode}')
        result = {}
        result['info'] = {}
        result['info']['type'] = 'jump'
        result['info']['mode'] = mode
        result['info']['layer'] = layer
        vecs = vectors[layer]
        if mode == 'mat' and layer != num_of_layers:
            if (layer, num_of_layers) not in As:
                As[(layer, num_of_layers)] =\
                    load_mat(model_folder_name, (layer, num_of_layers),
                             dataset=dataset)
            A = As[(layer, num_of_layers)]
            vecs = mul(A, vecs)
        prediction_dist = get_dist(vecs)
        prediction_topk = prediction_dist.topk(k=10, dim=1)[1]
        result['topk'] = prediction_topk
        e_to_minus_surprisal = trc.select_1_by_0(final_layer_dist,
                                                 prediction_topk[:, 0])
        surprisal = -torch.log(e_to_minus_surprisal)
        result['surprisal'] = surprisal
        pck.dump(result, save_file_name)

if compute_early_exit:

    params = {}
    params['minimal_layer'] = 1
    params['tau'] = 4
    params['N'] = N

    instruction_list = []
    for mode in ['id', 'mat']:
        lmbda_list = []
        lmbda_list.append(-1.112)
        lmbda_list += rng(0., 1., 10)
        lmbda_list.append(1.112)
        if mode == 'id':
            lmbda_list += rng(1., 1.112, 10, no_start=True, no_end=True)
        instruction_list += [(mode, lmbda) for lmbda in lmbda_list]

    n = vectors[0].size(0)

    for mode, lmbda in instruction_list:
        params['lambda'] = lmbda
        topks = []
        layers = []
        for idx in range(n):
            topk, layer = early_exit(idx, mode, params)
            topks.append(topk)
            layers.append(layer)
        topks = concat_iter(topks)
        result = {}
        result['info'] = {}
        result['info']['type'] = 'early_exit'
        result['info']['mode'] = mode
        result['info']['lambda'] = lmbda
        result['layers_processed'] = sum(layers) / len(layers)
        result['topk'] = topks

        pck.dump(result, save_file_name)
        verbose(f'finished mode={mode}, lambda={lmbda}')
