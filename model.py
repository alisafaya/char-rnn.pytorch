# https://github.com/spro/char-rnn.pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable

class Boom(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1, shortcut=False, output_size=512):
        super(Boom, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout) if dropout else None
        if not shortcut:
            self.linear2 = nn.Linear(dim_feedforward, output_size)
        self.shortcut = shortcut
        self.act = nn.GELU()

    def forward(self, input):
        x = self.act(self.linear1(input))
        if self.dropout: x = self.dropout(x)
        if self.shortcut:
            ninp = input.shape[-1]
            x = torch.narrow(x, -1, 0, x.shape[-1] // ninp * ninp)
            x = x.view(*x.shape[:-1], x.shape[-1] // ninp, ninp)
            z = x.sum(dim=-2)
        else:
            z = self.linear2(x)

        return z


class CharRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, charset, dropout=0.1, model="lstm", n_layers=2):
        super(CharRNN, self).__init__()
        self.charset = charset
        self.model = model.lower()
        self.input_size = len(charset)
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.encoder = nn.Embedding(self.input_size, embedding_size)
        self.eln = nn.LayerNorm(embedding_size, eps=1e-12)
        self.eboomer = Boom(embedding_size, dim_feedforward=embedding_size*4, dropout=dropout, shortcut=False, output_size=hidden_size)

        if self.model == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
        elif self.model == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=dropout)

        self.hln = nn.LayerNorm(hidden_size, eps=1e-12)
        self.hboomer = Boom(hidden_size, dim_feedforward=hidden_size*4, dropout=dropout, shortcut=True)
        self.decoder = nn.Linear(hidden_size, self.input_size)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        encoded = self.encoder(input)
        encoded = self.eln(encoded)
        encoded = self.eboomer(encoded)
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output = self.hln(output.view(batch_size, -1))
        output = self.hboomer(output)
        output = self.decoder(output)
        return output, hidden

    def forward2(self, input, hidden):
        encoded = self.encoder(input.view(1, -1))
        encoded = self.eln(encoded)
        encoded = self.eboomer(encoded)
        output, hidden = self.rnn(encoded.view(1, 1, -1), hidden)
        output = self.hln(output.view(1, -1))
        output = self.hboomer(output)
        output = self.decoder(output)
        return output, hidden

    def init_hidden(self, batch_size):
        if self.model == "lstm":
            return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
        return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))

    def embed(self, text): # one sample
        input = self.tokenize(text).view(-1, 1)
        encoded = self.encoder(input)
        encoded = self.eln(encoded)
        encoded = self.eboomer(encoded)
        output, hidden = self.rnn(encoded)
        output = self.hln(output)
        output = self.hboomer(output)
        return output.squeeze(1).unsqueeze(0) # Batchsize, Depth, Seq, Output

    def tokenize(self, text):
        tensor = torch.zeros(len(text)).long()
        for c in range(len(text)):
            try:
                tensor[c] = self.charset.index(text[c])
            except:
                continue
        return tensor