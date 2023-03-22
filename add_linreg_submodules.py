'''
Compute and save the linearly-regressed matrices approximating
a sub-module's output given its input
'''

import utils.pickle as pck
import utils.torch as trc

from aux import concat_iter, linreg, file, verbose

from deconstructed_GPT2 import DeconstructedGPT2, Rep

#############################

if __name__ != '__main__':
    raise RuntimeError('This script is not intended to be imported')

#############################

model_name = 'gpt2'
dataset = 'wikipedia'
batch_size = 4
final_device = 'cpu'

#############################

tokenized = pck.load(file('experiment', model_name,
                          dataset + '_tokenized_train', 'pickle'))
tokenized_sentences = tokenized['tokenized_sentences']
token_positions = tokenized['token_positions']

device = trc.get_cuda_if_possible()
if final_device == 'gpu':
    final_device = device
model = DeconstructedGPT2(model_name, dataset)
model._no_ln_f = True
model.to(device)

num_of_layers = model.num_of_layers()

v = Rep(tokenized_sentences,
        token_positions=token_positions,
        device=device)

verbose('finished loading')


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


instruction_list = []
for i in range(num_of_layers):
    instruction_list.append((jump_params(i+1, 'stop'), 'save output'))

for layer, output in\
    enumerate(
        model.forward_detailed_bh(v,
                                  what_to_return=['partial_registry'],
                                  final_device=final_device,
                                  batch_size=batch_size,
                                  instruction_list=instruction_list)
        ):
    verbose(f'working on layer {layer} out of {num_of_layers}')
    partial_registry =\
        concat_iter(otpt['partial_registry'] for otpt in output)
    for i in range(len(partial_registry)-1):
        linreg(partial_registry[i],
               partial_registry[i+1],
               intercept=False,
               file_name=file('linreg', model_name, dataset,
                              'p' + str(layer) + '_' + str(i) +
                              '_' + str(i+1), 'pickle'))
