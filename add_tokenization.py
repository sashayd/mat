'''
Compute and save tokenizations of sentences,
and a random token position in each sentence
'''

import torch
import random

import utils.pickle as pck

from aux import concat, get_tokenizer, file

#############################

if __name__ != '__main__':
    raise RuntimeError('This script is not intended to be imported')

#############################

sents_file_name = 'experiment/sentences/wikipedia_20K-sentences.pickle'
train_sents = [0, 9000]
val_sents = [9000, 12000]
max_token_seq_length = 1024
model_name = 'gpt2'
dataset = 'wikipedia'

#############################

train_save_file_name = file('experiment', model_name,
                            dataset + '_tokenized_train', 'pickle')
val_save_file_name = file('experiment', model_name,
                          dataset + '_tokenized_val', 'pickle')

tokenizer = get_tokenizer(model_name)

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


def save_tokenized(sentences, save_file_name):
    tokenized_sentences = []
    token_positions = None
    for i in range(len(sentences)):
        sentence = sentences[i]
        tokenized =\
            tokenizer(sentence,
                      return_tensors='pt',
                      truncation=truncation,
                      max_length=max_token_seq_length)['input_ids'][0, :]
        tokenized_sentences.append(tokenized)
        position = torch.tensor(random.randrange(len(tokenized))).unsqueeze(0)
        token_positions = concat(token_positions, position)
    result = {}
    result['tokenized_sentences'] = tokenized_sentences
    result['token_positions'] = token_positions
    pck.save(result, save_file_name)


if train_sents is not None:
    save_tokenized(lines[train_sents[0]:train_sents[1]], train_save_file_name)
if val_sents is not None:
    save_tokenized(lines[val_sents[0]:val_sents[1]], val_save_file_name)
