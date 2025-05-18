"""
This script evaluates and compares language models on perplexity.

It loads:
- A word-level Bigram model trained on GloVe embeddings.
- A character-level RNN (GRU-based) trained on raw text.

Then, it computes and prints perplexity scores on:
1. Shakespeare and Wikipedia datasets (plain text).
2. POS-tagged versions for the Bigram model.

Functions:
- load_bigram()         : Loads Bigram parameters and vocabulary.
- load_char_rnn()       : Loads the trained character-level RNN model.
- compute_bigram_perplexity() : Evaluates Bigram model perplexity.
- compute_char_rnn_perplexity(): Evaluates Char-RNN model perplexity.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from q1d_neural_lm import convert_to_lm_dataset
from q1d_neural_lm import load_vocab_embeddings
from q1c_neural import forward

from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
import torch

from data_utils import utils
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data/lm")
MODEL_DIR = Path("models")

VOCAB_PATH = DATA_DIR / "vocab.ptb.txt"
EMBEDDING_PATH = DATA_DIR / "vocab.embeddings.glove.txt"

BIGRAM_PATH = MODEL_DIR / "saved_params_40000.npy"
CHAR_RNN_PATH = MODEL_DIR / "char_rnn_model.pth"

EVAL_FILES = [
    "shakespeare_for_perplexity.txt",
    "wikipedia_for_perplexity.txt",
]

EVAL_POS_FILES = [
    "shakespare_pos_fromat.txt",
    "wikipedia_pos_fromat.txt",
]


class OurModel(nn.Module):
    """
       Simple character-level GRU model with embedding + linear output.
       The model was taken from q2 notebook.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(OurModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(num_embeddings=input_size,
                                      embedding_dim=hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, input_, hidden):
        if isinstance(input_, int):  # Python int
            input_ = torch.tensor([[input_]], dtype=torch.long, device=next(self.parameters()).device)
        elif isinstance(input_, torch.Tensor):
            if input_.dim() == 0:
                input_ = input_.view(1, 1)
            elif input_.dim() == 1:
                input_ = input_.view(1, -1)

        embedded = self.embedding(input_)  # (1, batch, hidden_size)
        output, hidden = self.gru(embedded, hidden)  # (1, batch, hidden_size)
        output = self.output_layer(output.squeeze(0))  # (batch, output_size)
        # -------------------------
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.num_layers, 1, self.hidden_size))


def load_bigram():
    """
       Loads the bigram model: vocabulary, embeddings, dimensions, parameters.

       :return: (params, dimensions, num_to_word_embedding, word_to_num)
    """
    vocab = pd.read_table(VOCAB_PATH,
                          header=None,
                          sep=r"\s+",
                          engine="python",
                          index_col=0,
                          names=['count', 'freq'])
    vocabsize = 2000
    num_to_word = dict(enumerate(vocab.index[:vocabsize]))
    word_to_num = utils.invert_dict(num_to_word)
    num_to_word_embedding = load_vocab_embeddings()

    input_dim = 50
    hidden_dim = 50
    output_dim = vocabsize
    dimensions = [input_dim, hidden_dim, output_dim]
    params = np.load(BIGRAM_PATH, allow_pickle=True)
    if hasattr(params, 'item'):
        # handle both np.save as dict and as array
        try:
            params = params.item()
        except Exception:
            pass

    return params, dimensions, num_to_word_embedding, word_to_num


def load_char_rnn():
    """
        Loads the character-level RNN model and vocabulary.

        :return: (char_rnn_model, stoi_dict)
    """
    checkpoint = torch.load(CHAR_RNN_PATH)
    stoi = checkpoint["stoi"]
    input_size = len(stoi)
    output_size = len(stoi)

    char_rnn = OurModel(input_size=input_size, hidden_size=100, output_size=output_size, num_layers=1)
    char_rnn.load_state_dict(checkpoint["state_dict"])
    char_rnn.eval()

    return char_rnn, stoi


def compute_bigram_perplexity(file_path,
                              params,
                              dimensions,
                              num_to_word_embedding,
                              word_to_num,
                              pos_tagged=False):
    """
       Computes perplexity for a word-level bigram model.

       :param file_path: path to input file
       :param params: model parameters
       :param dimensions: (input_dim, hidden_dim, output_dim)
       :param num_to_word_embedding: embedding vectors
       :param word_to_num: word-to-index dictionary
       :param pos_tagged: whether to strip POS tags
       :return: perplexity (float)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_docs = [line.strip().split() for line in f if line.strip()]

    if pos_tagged:
        docs = [[tok.rsplit('_', 1)[0] for tok in sent] for sent in raw_docs]
    else:
        docs = raw_docs

    # Convert to index sequences
    S_data = utils.docs_to_indices(docs, word_to_num)

    # Make bigram input/output pairs
    in_idx, out_idx = convert_to_lm_dataset(S_data)
    M = len(in_idx)

    # Sum log probabilities
    total_logprob = 0.0
    for src, tgt in zip(in_idx, out_idx):
        x = np.asarray(num_to_word_embedding[src]).reshape(1, dimensions[0])
        p = forward(x, tgt, params, dimensions)
        total_logprob += np.log(p + 1e-12)

    return np.exp(-total_logprob / M)


def compute_char_rnn_perplexity(file_path, model, stoi):
    """
       Computes perplexity for a character-level RNN model.

       :param file_path: path to input file
       :param model: trained character-level RNN model
       :param stoi: character-to-index dictionary
       :return: perplexity (float)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    text = text.strip()

    # Map each char to its index
    indices = [stoi[ch] for ch in text if ch in stoi]
    M = len(indices) - 1
    if M <= 0:
        raise ValueError("Not enough characters to compute perplexity.")

    # Iterate through the sequence, summing log-probs
    total_logprob = 0.0
    hidden = model.init_hidden()
    for i in range(M):
        inp = torch.tensor([[indices[i]]], dtype=torch.long)
        output, hidden = model(inp, hidden)
        probs = F.softmax(output, dim=-1).squeeze(0).detach().cpu().numpy()
        total_logprob += np.log(probs[indices[i + 1]] + 1e-12)

    # 4) Compute and return perplexity
    return np.exp(-total_logprob / M)

# Load models and data
char_rnn, stoi = load_char_rnn()
params, dimensions, num_to_word_embedding, word_to_num = load_bigram()

# Bigram perplexity
pp_bi_sh_pl = compute_bigram_perplexity(
    EVAL_FILES[0], params, dimensions, num_to_word_embedding, word_to_num, pos_tagged=False
)
pp_bi_wi_pl = compute_bigram_perplexity(
    EVAL_FILES[1], params, dimensions, num_to_word_embedding, word_to_num, pos_tagged=False
)

# Bigram on POS-tagged text
pp_bi_sh_pos = compute_bigram_perplexity(
    EVAL_POS_FILES[0], params, dimensions, num_to_word_embedding, word_to_num, pos_tagged=True
)
pp_bi_wi_pos = compute_bigram_perplexity(
    EVAL_POS_FILES[1], params, dimensions, num_to_word_embedding, word_to_num, pos_tagged=True
)

# Char-RNN on plain text
pp_cr_sh_pl = compute_char_rnn_perplexity(EVAL_FILES[0], char_rnn, stoi)
pp_cr_wi_pl = compute_char_rnn_perplexity(EVAL_FILES[1], char_rnn, stoi)

print(f"{'Model':<10s}  {'Passage':<12s}  {'Mode':<8s}  Perplexity")
print("-" * 45)
print(f"{'Bigram':<10s}  {'Shakespeare':<12s}  {'Plain':<8s}  {pp_bi_sh_pl:.2f}")
print(f"{'Bigram':<10s}  {'Wikipedia':<12s}   {'Plain':<8s}  {pp_bi_wi_pl:.2f}")
print(f"{'Bigram':<10s}  {'Shakespeare':<12s}  {'POS':<8s}    {pp_bi_sh_pos:.2f}")
print(f"{'Bigram':<10s}  {'Wikipedia':<12s}   {'POS':<8s}    {pp_bi_wi_pos:.2f}")
print(f"{'Char-RNN':<10s}  {'Shakespeare':<12s}  {'Plain':<8s}  {pp_cr_sh_pl:.2f}")
print(f"{'Char-RNN':<10s}  {'Wikipedia':<12s}   {'Plain':<8s}  {pp_cr_wi_pl:.2f}")
