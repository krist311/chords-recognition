import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from torch import autograd

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_size):
        super(LSTMClassifier, self).__init__()

        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1)

        self.hidden2out = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax()

        self.dropout_layer = nn.Dropout(p=0.2)

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
                autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)))

    def forward(self, batch):
        self.hidden = self.init_hidden(batch.size(-1))
        outputs, (ht, ct) = self.lstm(batch, self.hidden)

        # ht is the last hidden state of the sequences
        # ht = (1 x batch_size x hidden_dim)
        # ht[-1] = (batch_size x hidden_dim)
        output = self.dropout_layer(ht[-1])
        output = self.hidden2out(output)
        output = self.softmax(output)

        return output


class RandomForest(RandomForestClassifier):
    def __init__(self, ):
        super(RandomForest, self).__init__(criterion='entropy', max_features='log2', n_estimators=300)
