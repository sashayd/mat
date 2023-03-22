'''
Compute and save the linearly-regressed matrices approximating
a layer's representation given another layer's representation (BERT)
'''

from aux import load_vectors, linreg, file, verbose

#############################

if __name__ != '__main__':
    raise RuntimeError('This script is not intended to be imported')

#############################

model_folder_name = 'bert-base-uncased_mask'
dataset = 'wikipedia'
batch_size = 4
final_device = 'cpu'
only_to_final = False

#############################

vectors = load_vectors(model_folder_name, dataset + '_train')
num_of_layers = vectors['num_of_layers']

verbose('finished loading')

if only_to_final:
    list_of_layer_pairs = [(i, num_of_layers) for i in range(num_of_layers)]
else:
    list_of_layer_pairs = sum([[(i, j)
                                for j in range(i+1, num_of_layers+1)]
                              for i in range(num_of_layers)], [])

verbose('starting linear regression computation and save')
for i, j in list_of_layer_pairs:
    linreg(vectors[i], vectors[j],
           intercept=False,
           file_name=file('linreg', model_folder_name, dataset,
                          str(i) + '_' + str(j), 'pickle'))
