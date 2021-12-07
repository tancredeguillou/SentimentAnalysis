import torch
from torch import nn

class TextClassificationLinear(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationLinear, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


class TextClassificationMLP(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class, hidden_size1, hidden_size2, hidden_size3):
        super(TextClassificationMLP, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(embed_dim, hidden_size1)  # dense layer
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  # dense layer
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)  # dense layer
        self.fc4 = nn.Linear(hidden_size3, num_class)  # dense layer
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        self.fc3.weight.data.uniform_(-initrange, initrange)
        self.fc3.bias.data.zero_()
        self.fc4.weight.data.uniform_(-initrange, initrange)
        self.fc4.bias.data.zero_()


    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        h1 = self.fc1(embedded)
        a1 = self.activation(h1)
        h2 = self.fc2(a1)
        a2 = self.activation(h2)
        h3 = self.fc3(a2)
        a3 = self.activation(h3)
        h4 = self.fc4(a3)
        y = h4
        return y