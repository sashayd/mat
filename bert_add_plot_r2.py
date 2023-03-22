'''
Compute and save r^2 results, and plot a heatmap of them (BERT)
'''

import os
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt

import utils.pickle as pck

from aux import mul, load_vectors, file, verbose

#############################

if __name__ != '__main__':
    raise RuntimeError('This script is not intended to be imported')

#############################

model_folder_name = 'bert-base-uncased_mask'
dataset = 'wikipedia'
plot_both_together = True

#############################

save_file_name = file('experiment',
                      model_folder_name, dataset + '_r2_scores', 'pickle')


def compute_r2_scores():

    vectors = load_vectors(model_folder_name, dataset + '_val')
    num_of_layers = vectors['num_of_layers']

    verbose('starting r2 score computation and save')
    score_mat = {}
    for i in range(num_of_layers+1):
        for j in range(i+1, num_of_layers+1):
            x = vectors[i]
            y = vectors[j]
            file_name = file('linreg', model_folder_name, dataset,
                             '_'.join([str(i), str(j)]), 'pickle')
            A = pck.load(file_name)
            yhat = mul(A, x)
            score_mat[(i, j)] = r2_score(y, yhat)

    score_id = {}
    for i in range(num_of_layers+1):
        for j in range(i+1, num_of_layers+1):
            x = vectors[i]
            y = vectors[j]
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

sns.set(font_scale=2, rc={'figure.figsize': (15, 8.27)})

if not plot_both_together:
    ticks = [i for i in range(num_of_layers+1)]
    sns.heatmap(dct['score_mat'], annot=False, vmin=0., vmax=1.,
                xticklabels=ticks,
                yticklabels=ticks)

    plt.savefig(file('experiment', model_folder_name, 'plots', dataset,
                     'r2_scores_mat', 'pdf'))

    plt.clf()

    sns.heatmap(dct['score_id'], annot=False, vmin=0., vmax=1.,
                xticklabels=ticks,
                yticklabels=ticks)

    plt.savefig(file('experiment', model_folder_name, 'plots', dataset,
                     'r2_scores_id', 'pdf'))
else:
    cmap = 'seismic_r'

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ticks = [(i if i % 2 == 0 else None) for i in range(num_of_layers+1)]
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

    plt.savefig(file('experiment', model_folder_name, 'plots', dataset,
                     'r2_scores', 'pdf'),
                bbox_inches='tight', pad_inches=0)
