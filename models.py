import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier

torch.set_default_dtype(torch.float64)


class LSTMClassifier(nn.Module):
    # input_size hidden_size num_layers
    def __init__(self, input_size, hidden_dim, output_size, num_layers, use_gpu):
        super(LSTMClassifier, self).__init__()
        self.use_gpu = use_gpu
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.hidden2out = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def init_hidden(self, batch_size):
        if self.use_gpu:
            return (
                torch.zeros(self.num_layers, batch_size, self.hidden_dim, dtype=torch.float64).cuda(),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim, dtype=torch.float64).cuda())
        else:
            return (
                torch.zeros(self.num_layers, batch_size, self.hidden_dim, dtype=torch.float64),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim, dtype=torch.float64))

    def forward(self, batch):
        self.hidden = self.init_hidden(batch.size(0))
        outputs, self.hidden = self.lstm(batch, self.hidden)
        output = self.hidden2out(outputs)
        output = self.softmax(output)
        return output


class RandomForest(RandomForestClassifier):
    def __init__(self, criterion, max_features, n_estimators):
        super(RandomForest, self).__init__(criterion=criterion, max_features=max_features, n_estimators=n_estimators,
                                           warm_start=True)
