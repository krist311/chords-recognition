import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from torch import autograd
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)


class LSTMClassifier(nn.Module):
    # input_size hidden_size num_layers
    def __init__(self, input_size, hidden_dim, output_size, num_layers):
        super(LSTMClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=self.num_layers)

        self.hidden2out = nn.Linear(hidden_dim, output_size)
        # self.dropout_layer = nn.Dropout(p=0.2)

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim, dtype=torch.float64)),
                autograd.Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim, dtype=torch.float64)))

    def forward(self, batch):
        self.hidden = self.init_hidden(batch.size(0))
        # (seq_len, batch_size, input_size)
        batch = batch.view(batch.shape[1], batch.shape[0], batch.shape[2])
        outputs, self.hidden = self.lstm(batch, self.hidden)
        # output = self.dropout_layer(outputs)
        output = self.hidden2out(outputs)
        output = F.log_softmax(output, dim=2)
        output = output.view(output.size(1), output.size(2), output.size(0))
        return output


class RandomForest(RandomForestClassifier):
    def __init__(self):
        super(RandomForest, self).__init__(criterion='entropy', max_features='log2', n_estimators=300, warm_start=True)
