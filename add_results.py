'''
Compute and save results for the paper
'''

import torch

import utils.pickle as pck
import utils.torch as trc

from aux import concat_iter, file, verbose, rng

from deconstructed_GPT2 import DeconstructedGPT2, Rep

#############################

if __name__ != '__main__':
    raise RuntimeError('This script is not intended to be imported')

#############################

model_name = 'gpt2'
dataset = 'wikipedia'
batch_size = 4
final_device = 'cpu'

add_jumps = True
add_early_exit = True

full_jump_modes = ['idln',
                   'mat',
                   ('blk_mat', {'what_to_mat': set([1, 4])}),
                   ('blk_mat', {'what_to_mat': set([2])}),
                   ('blk_mat', {'what_to_mat': set([5])})
                   ]

full_early_exit_modes = ['idln',
                         'mat'
                         ]

jump_modes = full_jump_modes
early_exit_modes = full_early_exit_modes

#############################

save_file_name = file('experiment', model_name,
                      dataset + '_results', 'pickle')

tokenized = pck.load(file('experiment', model_name,
                          dataset + '_tokenized_val', 'pickle'))
tokenized_sentences = tokenized['tokenized_sentences']
token_positions = tokenized['token_positions']

# average length of tokenized sentence (up to our random token)
N = [token_positions[i]+1 for i in range(len(token_positions))]
N = sum(N) / len(N)
N = float(N)

device = trc.get_cuda_if_possible()
if final_device == 'gpu':
    final_device = device
model = DeconstructedGPT2(model_name, dataset)
model.to(device)

num_of_layers = model.num_of_layers()

v = Rep(tokenized_sentences,
        token_positions=token_positions,
        device=device)

verbose('finished loading...')

instruction_list = [({'mode': 'raw'}, 'output')]
output = next(
    model.forward_detailed_bh(v,
                              what_to_return=['output_softmax'], k=10,
                              final_device=final_device,
                              batch_size=batch_size,
                              instruction_list=instruction_list)
    )
raw_dist = concat_iter(otpt['output_softmax'] for otpt in output)
verbose('finished raw calculations...')


def jump_params(layer, jump_mode):
    params = {'mode': 'jump',
              'jump_layer': layer
              }
    if isinstance(jump_mode, str):
        params |= {'jump_mode': jump_mode}
    else:
        params |= {'jump_mode': jump_mode[0]}
        params |= jump_mode[1]
    return params


def compute_jumps(modes):
    modes = list(modes)

    what_to_return = ['topk']

    instruction_list = []
    for i in range(num_of_layers+1):
        instruction_list.append((jump_params(i, 'stop'), 'save'))
        for mode in modes:
            instruction_list.append((jump_params(i, mode), 'output'))

    mode_counter = 0
    layer = 0
    for output in model.forward_detailed_bh(v,
                                            what_to_return=what_to_return,
                                            k=10,
                                            final_device=final_device,
                                            batch_size=batch_size,
                                            instruction_list=instruction_list):
        mode = modes[mode_counter]
        result = {}
        result['info'] = {}
        result['info']['type'] = 'jump'
        result['info']['mode'] = mode
        result['info']['layer'] = layer
        topk = concat_iter(otpt['topk'] for otpt in output)
        e_to_minus_surprisal = trc.select_1_by_0(raw_dist.v(),
                                                 topk.v()[:, 0])
        surprisal = -torch.log(e_to_minus_surprisal)
        result['layers_processed'] = topk.layers_processed()
        result['topk'] = topk.v().to('cpu')
        result['surprisal'] = surprisal.to('cpu')
        pck.dump(result, save_file_name)
        mode_counter += 1
        if mode_counter == len(modes):
            mode_counter = 0
            verbose(f'finished layer {layer}')
            layer += 1


def early_exit_params(lmbda, ee_jump_mode):
    params = {'mode': 'early_exit',
              'minimal_layer': 1,
              'tau': 4.,
              'N': N,
              'lambda': lmbda
              }
    if isinstance(ee_jump_mode, str):
        params |= {'ee_jump_mode': ee_jump_mode}
    else:
        params |= {'ee_jump_mode': ee_jump_mode[0]}
        params |= ee_jump_mode[1]
    return params


def compute_early_exit(ee_jump_mode):

    lmbda_list = []
    lmbda_list.append(-1.112)
    lmbda_list += rng(0., 1., 10)
    lmbda_list.append(1.112)
    if ee_jump_mode == 'idln' or ee_jump_mode == 'id':
        lmbda_list += rng(1., 1.112, 10, no_start=True, no_end=True)

    instruction_list = []
    for lmbda in lmbda_list:
        instruction_list.append((early_exit_params(lmbda, ee_jump_mode),
                                 'output'))

    for c, output in\
        enumerate(model.forward_detailed_bh(v,
                                            what_to_return=['topk'], k=10,
                                            final_device=final_device,
                                            batch_size=1,
                                            instruction_list=instruction_list)
                  ):
        result = {}
        result['info'] = {}
        result['info']['type'] = 'early_exit'
        result['info']['mode'] = ee_jump_mode
        result['info']['lambda'] = lmbda_list[c]
        topk = concat_iter(otpt['topk'] for otpt in output)
        e_to_minus_surprisal = trc.select_1_by_0(raw_dist.v(),
                                                 topk.v()[:, 0])
        surprisal = -torch.log(e_to_minus_surprisal)
        result['layers_processed'] = topk.layers_processed()
        result['topk'] = topk.v().to('cpu')
        result['surprisal'] = surprisal.to('cpu')

        pck.dump(result, save_file_name)
        verbose(f'finished mode={ee_jump_mode}, lambda={lmbda_list[c]}')


if add_jumps:
    compute_jumps(jump_modes)

if add_early_exit:
    for ee_jump_mode in early_exit_modes:
        compute_early_exit(ee_jump_mode)
