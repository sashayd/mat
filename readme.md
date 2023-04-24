# Jump to Conclusions: Short-Cutting Transformers With Linear Transformations

This is the repository for the code used in the paper:

* Alexander Yom Din, Taelin Karidi, Leshem Choshen, Mor Geva. 2023. Jump to Conclusions: Short-Cutting Transformers With Linear Transformations. ([arXiv:2303.09435](https://arxiv.org/abs/2303.09435))

please cite the paper as:

```bibtex
@article{din2023jump,
      title={Jump to Conclusions: Short-Cutting Transformers With Linear Transformations},
      author={Yom Din, Alexander and Karidi, Taelin and Choshen, Leshem and Geva, Mor},
      journal={arXiv preprint arXiv:2303.09435},
      year={2023},
}
```

# Running the code

To produce plots for `gpt2` and Wikipedia sentences, run the following, in the written order:

- get_wikipedia_sentences.py

  (produces `./experiment/sentences/wikipedia_20K-sentences.pickle`, containing 20K sentences from Wikipedia)

- add_tokenization.py

  (produces `./experiment/gpt2/wikipedia_tokenized_train.pickle` containing the tokenizations and random token positions for the first 9000 sentences from the file produced by the previous script, and `./experiment/gpt2/wikipedia_tokenized_val.pickle` containing the tokenizations and random token positions for the next 3000 sentences)

- add_linreg.py

  (produces `./linreg/gpt2/wikipedia/i_j.pickle` where $0 \leq i < j \leq 12$, containing the matrix $A_{j,i}$ (as a torch.Tensor) used for skipping from layer $i$ to layer $j$)

- add_plot_r2.py

  (produces `./experiment/gpt2/wikipedia_r2_scores.pickle` containing the $r^2$ scores for $\texttt{mat}$ and $\texttt{id}$, and also produces `./experiments/gpt2/plots/wikipedia/r2_scores_12.pdf` containing the heatmap plots for these $r^2$ scores)

- add_linreg_submodules.py

  (produces `./linreg/gpt2/wikipedia/pi_a_b.pickle` where $0 \leq i < 12$ and $0 \leq a < 6$ and $b = a + 1$; these contain matrices (as a torch.Tensor) used to linearly approximate the output of a sub-module in transformer block $i+1$ given its input. $b=1$ corresponds to the first layer normalization, $b=2$ corresponds to attention, $b=3$ corresponds to the first residual connection, $b=4$ correspodns to the second layer normalization, $b=5$ corresponds to the feed-forward network layer and $b=6$ corresponds to the second residual connection)

- add_results.py

  (produces `./experiment/gpt2/wikipedia_results.pickle` containing (for each validation set sample) the top 10 tokens, as well as the model's surprisal of the top 1 token, according to the five mappings of the paper, at each layer; and also containing the top 10 tokens and number of layers processed when early-exiting and using the mappings $\texttt{mat}$ and $\texttt{id}$ (for various values of $\lambda$))

- plot_results.py

  (produces some plots in `./experiment/gpt2/plots/wikipedia/` based on the results in the previous file's output)

To produce plots for `bert-base-uncased` and Wikipedia sentences, run the following, in the written order:

- get_wikipedia_sentences.py

  (the same as for `gpt2` above, no need to re-run)

- bert_add_reps.py

  (produces `./experiment/bert-base-uncased_mask/wikipedia_train.pickle` containing the tokenizations, random token positions and representations of the masked random token at all layers for the first 9000 sentences from the file produced by the previous script, and `./experiment/bert-base-uncased_mask/wikipedia_val.pickle` containing the tokenizations, random token positions and representations of the masked random token at all layers for the next 3000 sentences)

- bert_add_linreg.py

  (produces `./linreg/bert-base-uncased_mask/wikipedia/i_j.pickle` where $0 \leq i < j \leq 12$, containing the matrix $A_{j,i}$ (as a torch.Tensor) used for skipping from layer $i$ to layer $j$)

- bert_add_plot_r2.py

  (produces `./experiment/bert-base-uncased_mask/wikipedia_r2_scores.pickle` containing the $r^2$ scores for $\texttt{mat}$ and $\texttt{id}$, and also produces `./experiments/bert-base-uncased_mask/plots/wikipedia/r2_scores_12.pdf` containing the heatmap plots for these $r^2$ scores)

- bert_add_results.py

  (produces `./experiment/bert-base-uncased_mask/wikipedia_results.pickle` containing (for each validation set sample) the top 10 tokens, as well as the model's surprisal of the top 1 token, according to $\texttt{mat}$ and $\texttt{id}$, at each layer; and also containing the top 10 tokens and number of layers processed when early-exiting and using the mappings $\texttt{mat}$ and $\texttt{id}$ (for various values of $\lambda$))

- plot_results.py (change `model_folder_name='bert-base-uncased_mask'` and `plot_parts = False`)

  (produces some plots in `./experiment/bert-base-uncased_mask/plots/wikipedia/` based on the results in the previous file's output)

We also produced plots for `gpt2-medium`, `gpt2-large`, `gpt2-xl`, `bert-large-uncased`. To do that, one should modify, in a relatively stratight-forward way, the variables at the head of each script in the sequence.

# Requirements

The code was ran with `Python 3.10.4` and the following package versions:

```
torch.__version__ = 1.13.1+cu117
transformers.__version__ = 4.20.1
sklearn.__version__ = 1.2.0
pickle.format_version = 4.0
datasets.__version__ = 2.5.2  # used only to fetch Wikipedia sentences
spacy.__version__ = 3.5.0  # used only to fetch Wikipedia sentences
```

# Trained Matrices

Some of the trained matrices can be found at [https://huggingface.co/sashay/linear-shortcut](https://huggingface.co/sashay/linear-shortcut).
