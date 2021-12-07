import torch
from torchtext.datasets import AG_NEWS
from torchtext.vocab import build_vocab_from_iterator

train_iter = AG_NEWS(split='train')

# Tokenize the data
from torchtext.data.utils import get_tokenizer
tokenizer = get_tokenizer('basic_english')

# Create vocabulary
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter),min_freq=20, specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Set up a pipeline
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1