import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F

torch.set_default_dtype(torch.float64)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

import torch


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv5x5(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=1, bias=False)


def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=1, bias=False)


class BasicBlock3x3(nn.Module):
    expansion = 1

    def __init__(self, inplanes3, planes, stride=1, downsample=None):
        super(BasicBlock3x3, self).__init__()
        self.conv1 = conv3x3(inplanes3, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock5x5(nn.Module):
    expansion = 1

    def __init__(self, inplanes5, planes, stride=1, downsample=None):
        super(BasicBlock5x5, self).__init__()
        self.conv1 = conv5x5(inplanes5, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv5x5(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:, :, 0:-d] + out
        out1 = self.relu(out1)
        # out += residual

        return out1


class BasicBlock7x7(nn.Module):
    expansion = 1

    def __init__(self, inplanes7, planes, stride=1, downsample=None):
        super(BasicBlock7x7, self).__init__()
        self.conv1 = conv7x7(inplanes7, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv7x7(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:, :, 0:-d] + out
        out1 = self.relu(out1)
        # out += residual

        return out1


class MSResNet(nn.Module):
    def __init__(self, input_channel, layers=[1, 1, 1, 1], num_classes=10):
        self.inplanes3 = 64
        self.inplanes5 = 64
        self.inplanes7 = 64

        super(MSResNet, self).__init__()

        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        self.layer3x3_1 = self._make_layer3(BasicBlock3x3, 64, layers[0], stride=2)
        self.layer3x3_2 = self._make_layer3(BasicBlock3x3, 128, layers[1], stride=2)
        self.layer3x3_3 = self._make_layer3(BasicBlock3x3, 256, layers[2], stride=2)
        # self.layer3x3_4 = self._make_layer3(BasicBlock3x3, 512, layers[3], stride=2)

        # maxplooing kernel size: 16, 11, 6
        self.maxpool3 = nn.AvgPool1d(kernel_size=11, stride=1, padding=0)

        self.layer5x5_1 = self._make_layer5(BasicBlock5x5, 64, layers[0], stride=2)
        self.layer5x5_2 = self._make_layer5(BasicBlock5x5, 128, layers[1], stride=2)
        self.layer5x5_3 = self._make_layer5(BasicBlock5x5, 256, layers[2], stride=2)
        # self.layer5x5_4 = self._make_layer5(BasicBlock5x5, 512, layers[3], stride=2)
        self.maxpool5 = nn.AvgPool1d(kernel_size=6, stride=1, padding=0)

        self.layer7x7_1 = self._make_layer7(BasicBlock7x7, 64, layers[0], stride=2)
        self.layer7x7_2 = self._make_layer7(BasicBlock7x7, 128, layers[1], stride=2)
        self.layer7x7_3 = self._make_layer7(BasicBlock7x7, 256, layers[2], stride=1)
        # self.layer7x7_4 = self._make_layer7(BasicBlock7x7, 512, layers[3], stride=2)
        self.maxpool7 = nn.AvgPool1d(kernel_size=4, stride=1, padding=0)

        # self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(256 * 3, num_classes)

        # todo: modify the initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm1d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer3(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes3, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes3, planes, stride, downsample))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes))

        return nn.Sequential(*layers)

    def _make_layer5(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes5 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes5, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes5, planes, stride, downsample))
        self.inplanes5 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes5, planes))

        return nn.Sequential(*layers)

    def _make_layer7(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes7 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes7, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes7, planes, stride, downsample))
        self.inplanes7 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes7, planes))

        return nn.Sequential(*layers)

    def forward(self, x0, lengths):
        batch_size = x0.size(0)
        seq_len = x0.size(1)
        x0 = x0.view(batch_size * seq_len, 1, x0.size(2))
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)

        x = self.layer3x3_1(x0)
        x = self.layer3x3_2(x)
        x = self.layer3x3_3(x)
        # x = self.layer3x3_4(x)
        x = self.maxpool3(x)

        y = self.layer5x5_1(x0)
        y = self.layer5x5_2(y)
        y = self.layer5x5_3(y)
        # y = self.layer5x5_4(y)
        y = self.maxpool5(y)

        z = self.layer7x7_1(x0)
        z = self.layer7x7_2(z)
        z = self.layer7x7_3(z)
        # z = self.layer7x7_4(z)
        z = self.maxpool7(z)

        out = torch.cat([x, y, z], dim=1)

        out = out.view(batch_size * seq_len, -1)
        # out = self.drop(out)
        out1 = self.fc(out)
        out = out.view(batch_size, seq_len, -1)
        out1 = out1.view(batch_size, seq_len, -1)

        return out1


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class AttentionLSTM(torch.nn.Module):

    def __init__(self, input_size, hidden_dim, output_size, num_layers, use_gpu, bidirectional,
                 dropout=(0.4, 0.0, 0.0)):
        super(AttentionLSTM, self).__init__()
        self.use_gpu = use_gpu
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.dropout1 = nn.Dropout(p=dropout[0])
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=self.num_layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout[1])
        # attention
        self.W1 = nn.Linear(self.num_directions * hidden_dim, self.num_directions*hidden_dim)
        self.W2 = nn.Linear(850, 850)
        self.V = nn.Linear(850, 850)
        ###self.hidden2out = nn.Linear(hidden_dim * self.num_directions, output_size)
        self.bn1 = nn.BatchNorm1d(hidden_dim*self.num_directions)
        self.dropout2 = nn.Dropout(p=dropout[2])
        self.hidden2out = nn.Linear(hidden_dim*self.num_directions, output_size)

    def disable_dropout(self):
        self.lstm.dropout = .0
        self.dropout1.p = .0
        self.dropout2.p = .0

    def init_hidden(self, batch_size):
        if self.use_gpu:
            return (
                torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim,
                            dtype=torch.float64).cuda(),
                torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim,
                            dtype=torch.float64).cuda())
        else:
            return (
                torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim, dtype=torch.float64),
                torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim, dtype=torch.float64))

    def attention_net(self, lstm_output, final_state):
        # if self.num_directions ==2:
        #     lstm_output = lstm_output[:,:,:self.hidden_dim].add(lstm_output[:,:,self.hidden_dim:])
        # lstm_output - [batch_size, seq_len, h_dim], attn_weights - [batch_size, seq_len, seq_len]
        # attn_weights = torch.bmm(lstm_output,lstm_output.transpose(1, 2))  # attn_weights - [batch_size, seq_len]
        # soft_attn_weights = F.softmax(attn_weights, 1)
        # new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights).transpose(1, 2)

        attn_weights = self.V(torch.tanh(torch.bmm(self.W1(lstm_output), self.W2(
            lstm_output.transpose(1, 2)))))  # attn_weights - [batch_size, seq_len]
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights).transpose(1, 2)

        return new_hidden_state

    def forward(self, batch, lengths):

        """
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for pos & neg class which receives its input as the new_hidden_state which is basically the output of the Attention network.
        final_output.shape = (batch_size, output_size)

        """
        self.hidden = self.init_hidden(batch.size(0))
        batch = self.dropout1(batch)
        # pack sequence if lengths available(during training)
        if lengths:
            batch = pack_padded_sequence(batch, lengths, batch_first=True)
        # final_hidden_state - [1, batch_size, hidden_dim]
        output, (final_hidden_state, final_cell_state) = self.lstm(batch, self.hidden)
        if lengths:
            output, _ = pad_packed_sequence(output, batch_first=True)
        # [batch_size,seq_len, hidden_dim]
        attn_output = self.attention_net(output, final_hidden_state)
        # attn_output = self.attention_layer(output)
        output = self.bn1(attn_output.permute(0, 2, 1)).permute(0, 2, 1)
        output = self.dropout2(output)
        output = self.hidden2out(output)
        return output


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size, num_layers, use_gpu, bidirectional,
                 dropout=(0.4, 0.0, 0.0)):
        super(LSTMClassifier, self).__init__()
        self.use_gpu = use_gpu
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.dropout1 = nn.Dropout(p=dropout[0])
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=self.num_layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout[1])
        self.bn1 = nn.BatchNorm1d(hidden_dim * self.num_directions)
        self.dropout2 = nn.Dropout(p=dropout[2])
        self.hidden2out = nn.Linear(hidden_dim * self.num_directions, output_size)

    def disable_dropout(self):
        self.lstm.dropout = .0
        self.dropout1.p = .0
        self.dropout2.p = .0

    def init_hidden(self, batch_size):
        if torch.cuda.is_available():
            return (
                torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim,
                            dtype=torch.float64).cuda(),
                torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim,
                            dtype=torch.float64).cuda())
        else:
            return (
                torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim, dtype=torch.float64),
                torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim, dtype=torch.float64))

    def forward(self, batch, lengths=None):
        self.hidden = self.init_hidden(batch.size(0))
        batch = self.dropout1(batch)
        # pack sequence if lengths available(during training)
        if lengths:
            batch = pack_padded_sequence(batch, lengths, batch_first=True)
        ####hidden - [1, batch_size, hidden_dim]
        output, self.hidden = self.lstm(batch, self.hidden)
        if lengths:
            output, _ = pad_packed_sequence(output, batch_first=True)
        # [batch_size,seq_len, hidden_dim]
        output = self.bn1(output.permute(0, 2, 1)).permute(0, 2, 1)
        output = self.dropout2(output)
        output = self.hidden2out(output)
        return output


class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size, num_layers, use_gpu, bidirectional,
                 dropout=(0.4, 0.0, 0.0)):
        super(GRUClassifier, self).__init__()
        self.use_gpu = use_gpu
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.dropout1 = nn.Dropout(p=dropout[0])
        self.gru = nn.GRU(input_size, hidden_dim, num_layers=self.num_layers, batch_first=True,
                          bidirectional=bidirectional, dropout=dropout[1])
        self.bn1 = nn.BatchNorm1d(hidden_dim * self.num_directions)
        self.hidden2out = nn.Linear(hidden_dim * self.num_directions, output_size)
        self.dropout2 = nn.Dropout(p=dropout[2])

    def disable_dropout(self):
        self.gru.dropout = .0
        self.dropout1.p = .0
        self.dropout2.p = .0

    def init_hidden(self, batch_size):
        if self.use_gpu:
            return (
                torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim,
                            dtype=torch.float64).cuda())
        else:
            return (
                torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim, dtype=torch.float64))

    def forward(self, batch, lengths=None):
        self.hidden = self.init_hidden(batch.size(0))
        batch = self.dropout1(batch)
        # pack sequence if lengths available(during training)
        if lengths:
            batch = pack_padded_sequence(batch, lengths, batch_first=True)
        ####
        output, self.hidden = self.gru(batch, self.hidden)
        if lengths:
            output, _ = pad_packed_sequence(output, batch_first=True)
        output = self.bn1(output.permute(0, 2, 1)).permute(0, 2, 1)
        output = self.dropout2(output)
        output = self.hidden2out(output)
        return output


class RandomForest(RandomForestClassifier):
    def __init__(self, criterion, max_features, n_estimators):
        super(RandomForest, self).__init__(criterion=criterion, max_features=max_features, n_estimators=n_estimators,
                                           warm_start=True, n_jobs=-1)
