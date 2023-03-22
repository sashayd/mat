'''
Save plots of the results
'''

import torch

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import utils.pickle as pck

from aux import file

#############################

if __name__ != '__main__':
    raise RuntimeError('This script is not intended to be imported')

#############################

model_folder_name = 'gpt2'
num_of_layers = 12
dataset = 'wikipedia'

plot_jump = True
plot_parts = True
plot_ee = True

colors = {'id': (1., 0., 0.),
          'mat': (0., 0., 1.),
          'mat_attn': (0., 1., 0.),
          'mat_ffn': (1., 0., 1.),
          'mat_ln1_ln2': (1., 0.5, 0.)}

#############################


def plot_precision(df,
                   types, modes, precs,
                   hue=None, style=None, size=None,
                   hue_order=None, style_order=None, size_order=None,
                   errorbar=None,
                   xticks_every=None,
                   save_file_name=None):
    precs = ['precision@' + str(prec) for prec in precs]

    df = df.loc[((df['type'].isin(types)) &
                 df['mode'].isin(modes) &
                 (df['cols'].isin(precs)))]

    df = df.replace('precision@1', 'k=1')
    df = df.replace('precision@5', 'k=5')
    df = df.replace('precision@10', 'k=10')
    precs = ['k=' + prec.split('@')[1] for prec in precs]

    setting_dict = {}
    for setting in ['hue_order', 'style_order', 'size_order']:
        if locals()[setting] is None:
            setting_dict[setting] = None
        else:
            setting_dict[setting] = locals()[locals()[setting]]

    g = sns.relplot(data=df,
                    hue_order=setting_dict['hue_order'],
                    style_order=setting_dict['style_order'],
                    size_order=setting_dict['size_order'],
                    x='layers_processed',
                    y='vals',
                    palette=sns.color_palette([colors[m] for m in modes]),
                    kind='line',
                    errorbar=errorbar,
                    hue=hue,
                    style=style,
                    size=size)

    if 'early_exit' in types:
        xlabel = 'layers processed'
    else:
        xlabel = 'layer'

    if len(precs) == 1:
        ylabel = 'Precision@' + precs[0].split('=')[1]
    else:
        ylabel = 'Precision@k'

    if xticks_every is None:
        xticks_every = 2

    g.set(xlim=(0, num_of_layers),
          xticks=[xticks_every*i for i in
                  range(num_of_layers//xticks_every+1)],
          ylabel=ylabel,
          xlabel=xlabel)

    g.legend.set_title(None)
    for text in g.legend.texts:
        if text.get_text() in ['mode', 'type', 'cols']:
            text.set_text('')
        if text.get_text() == 'early_exit':
            text.set_text('early exit')
        if text.get_text() == 'jump':
            text.set_text('fixed exit')
    if g.legend.texts[0].get_text() == '':
        g.legend.texts = g.legend.texts[1:]
        g.legend.legendHandles = g.legend.legendHandles[1:]
    sns.move_legend(g, 'upper right')

    if save_file_name is not None:
        save_file_name =\
            file('experiment', model_folder_name,
                 'plots', dataset,
                 save_file_name, 'pdf')
        plt.savefig(save_file_name,
                    bbox_inches='tight')
    plt.clf()


def plot_surprisal(df,
                   types, modes,
                   hue=None, style=None, size=None,
                   hue_order=None, style_order=None, size_order=None,
                   col_name='surprisal',
                   ylim=None,
                   xticks_every=None,
                   save_file_name=None):
    df = df.loc[((df['type'].isin(types)) &
                 df['mode'].isin(modes) &
                 (df['cols'] == col_name))]
    setting_dict = {}
    for setting in ['hue_order', 'style_order', 'size_order']:
        if locals()[setting] is None:
            setting_dict[setting] = None
        else:
            setting_dict[setting] = locals()[locals()[setting]]
    g = sns.relplot(data=df,
                    hue_order=setting_dict['hue_order'],
                    style_order=setting_dict['style_order'],
                    size_order=setting_dict['size_order'],
                    x='layers_processed',
                    y='vals',
                    palette=sns.color_palette([colors[m] for m in modes]),
                    kind='line',
                    hue=hue,
                    style=style,
                    size=size)

    if 'early_exit' in types:
        xlabel = 'layers processed'
    else:
        xlabel = 'layer'

    if xticks_every is None:
        xticks_every = 2

    g.set(xlim=(0, num_of_layers),
          xticks=[xticks_every*i for i in
                  range(num_of_layers//xticks_every+1)],
          ylabel='Surprisal',
          xlabel=xlabel)
    if ylim is not None:
        g.set(ylim=ylim)

    g.legend.set_title(None)
    for text in g.legend.texts:
        if text.get_text() in ['mode', 'type', 'cols']:
            text.set_text('')
        if text.get_text() == 'early_exit':
            text.set_text('early exit')
        if text.get_text() == 'jump':
            text.set_text('fixed exit')
    if g.legend.texts[0].get_text() == '':
        g.legend.texts = g.legend.texts[1:]
        g.legend.legendHandles = g.legend.legendHandles[1:]
    sns.move_legend(g, 'upper right')

    if save_file_name is not None:
        save_file_name =\
            file('experiment', model_folder_name,
                 'plots', dataset,
                 save_file_name, 'pdf')
        plt.savefig(save_file_name,
                    bbox_inches='tight')
    plt.clf()

#############################


data = pck.load_all(file('experiment', model_folder_name,
                         dataset + '_results', 'pickle'))
data = list(data)

# set final_topk to be the topk of the final layer
for d in data:
    if d['info']['type'] != 'jump':
        continue
    if d['info']['layer'] != num_of_layers:
        continue
    final_topk = d['topk']

# set n to be number of sentence samples
for d in data:
    if 'topk' in d:
        n = d['topk'].size()[0]
        break

# setup data frame
df = pd.DataFrame()
for d in data:
    typ = d['info']['type']

    if typ not in ['jump', 'early_exit']:
        continue

    if 'mode' in d['info']:
        mode = d['info']['mode']
        if isinstance(mode, tuple):
            if mode[0] != 'blk_mat':
                assert RuntimeError(f'mode[0] = {mode[0]}')
            what = mode[1]['what_to_mat']
            if what == set([2]):
                mode = 'mat_attn'
            elif what == set([1, 4]):
                mode = 'mat_ln1_ln2'
            elif what == set([5]):
                mode = 'mat_ffn'
            else:
                assert RuntimeError(f'mode[2][what_to_mat] = {what}')
        elif isinstance(mode, str):
            if mode == 'idln':
                mode = 'id'
    else:
        mode = None

    dct = {}
    dct['type'] = typ
    dct['mode'] = mode

    if typ == 'early_exit':
        dct['identifier'] = d['info']['lambda']
        dct['layers_processed'] = d['layers_processed']
    elif typ == 'jump':
        dct['identifier'] = d['info']['layer']
        dct['layers_processed'] = float(d['info']['layer'])

    dct['sentence'] = [i for i in range(n)]
    if 'surprisal' in d:
        dct['surprisal'] =\
            [d['surprisal'][i].item()
             for i in range(n)]
    if 'topk' in d:
        for prec in [1, 5, 10]:
            dct['precision@' + str(prec)] =\
                [int(torch.isin(d['topk'][i, 0], final_topk[i, :prec]).item())
                 for i in range(n)]

    df = pd.concat([df, pd.DataFrame(dct)])

df = df.melt(['sentence',
              'type',
              'identifier',
              'layers_processed',
              'mode'],
             var_name='cols', value_name='vals')

sns.set(font_scale=1.7)

if num_of_layers == 12:
    xticks_every = 2
elif num_of_layers == 24:
    xticks_every = 3
elif num_of_layers == 36:
    xticks_every = 4
elif num_of_layers == 48:
    xticks_every = 6
else:
    xticks_every = num_of_layers // 12

if plot_jump:
    plot_precision(df,
                   types=['jump'],
                   modes=['mat', 'id'],
                   precs=[1, 5, 10],
                   hue='mode',
                   style='mode',
                   size='cols',
                   hue_order='modes',
                   style_order='modes',
                   size_order='precs',
                   xticks_every=xticks_every,
                   save_file_name='pre_' + str(num_of_layers))
    plot_surprisal(df,
                   types=['jump'],
                   modes=['mat', 'id'],
                   hue='mode',
                   style='mode',
                   hue_order='modes',
                   style_order='modes',
                   xticks_every=xticks_every,
                   save_file_name='surp_' + str(num_of_layers))
    # print the actual values
    print('(jump values)')
    df1 = df.loc[((df['type'] == 'jump') &
                 (df['cols'] == 'precision@1'))]
    df2 = df1.groupby(['mode', 'identifier'])
    df3 = df2.mean()[['layers_processed', 'vals']]
    print(df3.to_string())
    id_lst = df3[df3.index.get_level_values(0) == 'id']['vals']
    mat_lst = df3[df3.index.get_level_values(0) == 'mat']['vals']
    print('(differences)')
    print([mat_lst[i] - id_lst[i] for i in range(len(id_lst))])

if plot_parts:
    for prec in [1, 5, 10]:
        plot_precision(df,
                       types=['jump'],
                       modes=['mat_attn', 'mat_ln1_ln2', 'mat_ffn', 'mat'],
                       precs=[prec],
                       hue='mode',
                       style='mode',
                       hue_order='modes',
                       style_order='modes',
                       xticks_every=xticks_every,
                       save_file_name=('parts_pre' + str(prec) +
                                       '_' + str(num_of_layers)))

    plot_surprisal(df,
                   types=['jump'],
                   modes=['mat_attn', 'mat_ln1_ln2', 'mat_ffn', 'mat'],
                   hue='mode',
                   style='mode',
                   hue_order='modes',
                   style_order='modes',
                   xticks_every=xticks_every,
                   save_file_name='parts_surp_' + str(num_of_layers))

if plot_ee:
    plot_precision(df,
                   types=['early_exit', 'jump'],
                   modes=['mat', 'id'],
                   precs=[1],
                   hue='mode',
                   style='mode',
                   size='type',
                   hue_order='modes',
                   style_order='modes',
                   size_order='types',
                   xticks_every=xticks_every,
                   save_file_name='ee_pre1_' + str(num_of_layers))
    # print the actual values
    print('(early exit values)')
    df1 = df.loc[((df['type'] == 'early_exit') &
                 (df['cols'] == 'precision@1'))]
    df2 = df1.groupby(['mode', 'identifier'])
    print(df2.mean()[['layers_processed', 'vals']].to_string())
