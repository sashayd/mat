'''
Extract sentences from Wikipedia
'''

from datasets import load_dataset
import spacy
import numpy as np

import utils.pickle as pck
from aux import file

#############################

if __name__ != '__main__':
    raise RuntimeError('This script is not intended to be imported')

#############################

num_of_sents_to_save = 20000
save_file_name = file('experiment', 'sentences',
                      'wikipedia_20K-sentences', 'pickle')

#############################

wikipedia = load_dataset('wikipedia', '20220301.en')['train']
nlp = spacy.load('en_core_web_sm')

num_of_docs = len(wikipedia)


def get_sents(doc_id):
    spacyfied_doc = nlp(wikipedia[doc_id]['text'])
    return [sent for sent in spacyfied_doc.sents]

#############################


doc_ids_already_used = set()

for i in range(num_of_sents_to_save):
    stop = False
    while not stop:
        while True:
            doc_id = int(np.random.randint(0, num_of_docs))
            if doc_id not in doc_ids_already_used:
                break
        sents = get_sents(doc_id)
        for counter in range(10):
            rand_sent = int(np.random.randint(0, len(sents)))
            sent = str(sents[rand_sent])
            if sent[-1] == '\n':
                pck.dump(sent, save_file_name)
                stop = True
                break
        if stop:
            break
    doc_ids_already_used.add(doc_id)
    if i % 50 == 49:
        print(f'finished {i+1} sentences.')
