'''
Compute and save r^2 results, and plot a heatmap of them
'''

import os
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt

import utils.pickle as pck
import utils.torch as trc

from aux import mul, concat_iter, file, verbose

from deconstructed_GPT2 import DeconstructedGPT2, Rep

#############################

if __name__ != '__main__':
    raise RuntimeError('This script is not intended to be imported')

#############################

model_name = 'gpt2'
dataset = 'wikipedia'
batch_size = 8
final_device = 'cpu'
plot_both_together = True

#############################

save_file_name = file('experiment',
                      model_name, dataset + '_r2_scores', 'pickle')


def compute_r2_scores():
    global final_device

    tokenized = pck.load(file('experiment', model_name,
                              dataset + '_tokenized_val', 'pickle'))
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
    for i in range(num_of_layers+1):
        instruction_list.append((jump_params(i, 'stop'), 'save output'))

    vectors = []
    for layer, output in\
        enumerate(
            model.forward_detailed_bh(v,
                                      what_to_return=['output'],
                                      final_device=final_device,
                                      batch_size=batch_size,
                                      instruction_list=instruction_list)
            ):
        verbose(f'working on layer {layer} out of {num_of_layers}')
        vectors.append(concat_iter(otpt['output'] for otpt in output))

    verbose('starting r2 score computation and save')
    score_mat = {}
    for i in range(num_of_layers+1):
        for j in range(i+1, num_of_layers+1):
            x = vectors[i].v().detach().cpu()
            y = vectors[j].v().detach().cpu()
            file_name = file('linreg', model_name, dataset,
                             '_'.join([str(i), str(j)]), 'pickle')
            A = pck.load(file_name)
            yhat = mul(A, x)
            score_mat[(i, j)] = r2_score(y, yhat)

    score_id = {}
    for i in range(num_of_layers+1):
        for j in range(i+1, num_of_layers+1):
            x = vectors[i].v().detach().cpu()
            y = vectors[j].v().detach().cpu()
            yhat = x
            score_id[(i, j)] = r2_score(y, yhat)

    dct = {'score_mat': score_mat, 'score_id': score_id}

    pck.save(dct, save_file_name)
    return dct


if not os.path.isfile(save_file_name):
    dct = compute_r2_scores()
else:
    dct = pck.load(save_file_name)

num_of_layers = max([j for (i, j) in dct['score_id']])

for key in list(dct.keys()):
    for i in range(num_of_layers+1):
        for j in range(i+1):
            if i == j:
                dct[key][(i, j)] = 1.
            else:
                dct[key][(i, j)] = float('nan')
    dct[key] = [[dct[key][(i, j)]
                 for j in range(num_of_layers+1)]
                for i in range(num_of_layers+1)]

# sns.set(font_scale=1.3, rc={'figure.figsize': (11.7, 8.27)})
sns.set(font_scale=2, rc={'figure.figsize': (15, 8.27)})

if not plot_both_together:
    ticks = [i for i in range(num_of_layers+1)]
    sns.heatmap(dct['score_mat'], annot=False, vmin=0., vmax=1.,
                xticklabels=ticks,
                yticklabels=ticks)

    plt.savefig(file('experiment', model_name, 'plots', dataset,
                     'r2_scores_mat', 'pdf'))
    plt.clf()

    sns.heatmap(dct['score_id'], annot=False, vmin=0., vmax=1.,
                xticklabels=ticks,
                yticklabels=ticks)

    plt.savefig(file('experiment', model_name, 'plots', dataset,
                     'r2_scores_id', 'pdf'))
else:
    cmap = 'seismic_r'

    if num_of_layers == 48:
        ticks_every = 4
    else:
        ticks_every = 2

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ticks = [(i if i % ticks_every == 0 else None)
             for i in range(num_of_layers+1)]
    cbar_ax = fig.add_axes([.91, 0.1, .03, 0.8])
    mat_map = sns.heatmap(dct['score_mat'], annot=False, vmin=0., vmax=1.,
                          cmap=cmap,
                          xticklabels=ticks,
                          yticklabels=ticks,
                          cbar=False,
                          ax=ax1)
    id_map = sns.heatmap(dct['score_id'], annot=False, vmin=0., vmax=1.,
                         cmap=cmap,
                         xticklabels=ticks,
                         yticklabels=ticks,
                         cbar=True,
                         cbar_ax=cbar_ax,
                         ax=ax2)

    mat_map.set_ylabel('\u2113', fontsize=35)
    mat_map.set_xlabel('\u2113\'', fontsize=35)
    id_map.set_ylabel('\u2113', fontsize=35)
    id_map.set_xlabel('\u2113\'', fontsize=35)
    mat_map.set(title='mat')
    id_map.set(title='id')

    ticks = list(ax2.collections[0].colorbar.get_ticks())
    ticks = [f'{t:.1f}' for t in ticks]
    ticks[0] = '\u2264' + '0.0'
    ax2.collections[0].colorbar.set_ticklabels(ticks)

    plt.savefig(file('experiment', model_name, 'plots', dataset,
                     'r2_scores_' + str(num_of_layers), 'pdf'),
                bbox_inches='tight', pad_inches=0)
