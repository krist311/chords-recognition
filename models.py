import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from torch import autograd
import torch.nn.functional as F


from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
torch.set_default_dtype(torch.float64)

class LSTMClassifier(nn.Module):
    # input_size hidden_size num_layers
    def __init__(self, input_size, hidden_dim, output_size):
        super(LSTMClassifier, self).__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=1)

        self.hidden2out = nn.Linear(hidden_dim, output_size)

        self.dropout_layer = nn.Dropout(p=0.2)

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.randn(1, batch_size, self.hidden_dim, dtype=torch.float64)),
                autograd.Variable(torch.randn(1, batch_size, self.hidden_dim, dtype=torch.float64)))

    def forward(self, batch):
        # (seq_len, batch_size, input_size)
        batch = batch.view(batch.shape[1], batch.shape[0], batch.shape[2])
        self.hidden = self.init_hidden(batch.size(1))
        outputs, self.hidden = self.lstm(batch, self.hidden)

        # ht is the last hidden state of the sequences
        # ht = (1 x batch_size x hidden_dim)
        # ht[-1] = (batch_size x hidden_dim)
        output = self.dropout_layer(outputs)
        output = self.hidden2out(output)
        output = F.log_softmax(output, dim=1)
        output = output.view(output.size(1), output.size(2), output.size(0))
        return output


class RandomForest(RandomForestClassifier):
    def __init__(self, ):
        super(RandomForest, self).__init__(criterion='entropy', max_features='log2', n_estimators=300)
