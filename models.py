import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F

torch.set_default_dtype(torch.float64)


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
        #attention
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(850, 850)
        self.V = nn.Linear(850, 850)
        ###self.hidden2out = nn.Linear(hidden_dim * self.num_directions, output_size)
        self.bn1 = nn.BatchNorm1d(hidden_dim * self.num_directions)
        self.dropout2 = nn.Dropout(p=dropout[2])
        self.hidden2out = nn.Linear(hidden_dim * self.num_directions, output_size)

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
        # #[batch_size, seq_len, h_dim]
        # w1_lstm_output = self.W1(lstm_output)
        # w2_lstm_output = self.W2(lstm_output)
        #
        # #iterate over seq len
        # score = torch.zeros((lstm_output.size()[0],lstm_output.size()[1],lstm_output.size()[1]))
        # for hidden_ind in range(w2_lstm_output.size()[1]):
        #     #[batch_size, h_dim)
        #     hidden_with_time_axis = w2_lstm_output[:,hidden_ind,:].unsqueeze(1)
        #
        #     # score shape == (batch_size, max_length, 1)
        #     # we get 1 at the last axis because we are applying tanh(FC(EO) + FC(H)) to self.V
        #     # this is the step 1 described in the blog to compute scores s1, s2, ...
        #     cur_score = self.V(w1_lstm_output + hidden_with_time_axis).squeeze(2)
        #     score[:,:,hidden_ind] = cur_score
        #
        #     # attention_weights shape == (batch_size, max_length, 1)
        #     # this is the step 2 described in the blog to compute attention weights e1, e2, ...
        #
        # attention_weights = F.softmax(score, axis=1)
        #
        #
        # # context_vector shape after sum == (batch_size, hidden_size)
        # # this is the step 3 described in the blog to compute the context_vector = e1*h1 + e2*h2 + ...
        # #context_vector = attention_weights * enc_output
        # #context_vector = tf.reduce_sum(context_vector, axis=1)
        #
        # # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        # #x = self.embedding(x)
        #
        # # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        # # this is the step 4 described in the blog to concatenate the context vector with the output of the previous time step
        # #x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        #
        # # passing the concatenated vector to the GRU
        #
        # # output shape == (batch_size * 1, hidden_size)
        # #output = tf.reshape(output, (-1, output.shape[2]))
        #
        # # output shape == (batch_size * 1, vocab)
        # # this is the step 5 in the blog, to compute the next output word in the sequence
        # x = self.fc(output)
        #
        #
        #
        #
        #
        #
        #lstm_output - [batch_size, seq_len, h_dim], attn_weights - [batch_size, seq_len, seq_len]
        attn_weights = self.V(torch.tanh(torch.bmm(self.W1(lstm_output), self.W2(lstm_output.transpose(1, 2)))))  # attn_weights - [batch_size, seq_len]
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
        #attn_output = self.attention_layer(output)
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
