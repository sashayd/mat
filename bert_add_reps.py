'''
Compute and save layer representations (BERT)
'''

import torch

import utils.pickle as pck
import utils.torch as trc

from aux import file, verbose, get_tokenizer, get_model, randomize_below

#############################

if __name__ != '__main__':
    raise RuntimeError('This script is not intended to be imported')

#############################


sents_file_name = 'experiment/sentences/wikipedia_20K-sentences.pickle'
train_sents = [0, 9000]
val_sents = [9000, 12000]
model_name = 'bert-base-uncased'
dataset = 'wikipedia'
max_token_seq_length = 512
batch_size = 16
do_mask = True

#############################

mask_file_name_addition = '_mask' if do_mask else ''
train_save_file_name = file('experiment', model_name + mask_file_name_addition,
                            dataset + '_train', 'pickle')
val_save_file_name = file('experiment', model_name + mask_file_name_addition,
                          dataset + '_val', 'pickle')

verbose(f'Loading {model_name} (from transformers) '
        'model and tokenizer...')

tokenizer = get_tokenizer(model_name)
model = get_model(model_name)

if do_mask:
    mask_token = tokenizer.convert_tokens_to_ids('[MASK]')
    verbose(f'(the [MASK] token has token id {mask_token})')

device = trc.get_cuda_if_possible()
verbose('moving model to device (if not cpu)...')
model.to(device)

verbose(f'Loading sentences from {sents_file_name}...')

if sents_file_name.endswith('txt'):
    with open(sents_file_name, 'r') as f:
        lines = f.read().splitlines()
elif sents_file_name.endswith('pickle'):
    lines = [sent for sent in pck.load_all(sents_file_name)]
else:
    raise RuntimeError('sentences file name should be .txt or .pickle')

if max_token_seq_length is not None:
    truncation = True
else:
    truncation = False

n_train = train_sents[1] - train_sents[0]
n_val = val_sents[1] - val_sents[0]
n = n_train + n_val

verbose(f'Loaded {len(lines)} sentences, will use {n} sentences, '
        f'first {n_train} for training and the rest for validation.')

with torch.no_grad():
    i = 0
    counter = 0
    while i < n:
        if i < n_train:
            i1 = min(i + batch_size, n_train)
        else:
            i1 = min(i + batch_size, n)
        sents = lines[i:i1]
        tokenized = tokenizer(sents, return_tensors='pt', padding=True,
                              truncation=truncation,
                              max_length=max_token_seq_length)
        lengths = tokenized['attention_mask'].sum(dim=1)

        indices = randomize_below(lengths)

        # for BERT [MASK] experiment, we replace the random token
        # with a [MASK] token
        if do_mask:
            for j in range(len(sents)):
                tokenized['input_ids'][j, indices[j]] = mask_token

        tokenized = tokenized.to(device)

        outputs = model(**tokenized).hidden_states

        vectors = {}
        tokenized_sentences = tokenized['input_ids'].detach().cpu()
        tokenized_sentences = [tokenized_sentences[j, :lengths[j]]
                               for j in range(i1-i)]
        vectors['tokenized_sentences'] = tokenized_sentences
        vectors['token_positions'] = indices.detach()
        indices = indices.to(device)
        for layer in range(len(outputs)):
            vectors[layer] = trc.select_1_by_0(outputs[layer], indices)
            vectors[layer] = vectors[layer].detach().cpu()
        if i < n_train:
            pck.dump(vectors, train_save_file_name)
        else:
            pck.dump(vectors, val_save_file_name)
        if counter % 10 == 0:
            verbose(f'Finished {i1} sentences.')
        i = i1
        counter += 1

verbose(f'Saved results to {train_save_file_name} '
        f'and {val_save_file_name}.')
