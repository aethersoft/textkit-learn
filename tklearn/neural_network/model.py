import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class TextCNN(nn.Module):
    """Multichannel Convolutional Neural Network for Text Data.

    For more information please look at
    `Kim, Y. (2014). Convolutional neural networks for sentence classification. arXiv preprint arXiv:1408.5882.`
    """

    DEFAULT_CONFIG = {
        'model': 'rand',  # ['rand', 'static', 'non-static', 'multichannel']
        'max_sent_len': 20,
        'vocab_size': None,
        'word_dim': None,
        'class_size': 2,
        'filters': [3, 4, 5],
        'filter_num': 100,
        'dropout_prob': 0.5,
        'wv_matrix': None,
    }

    def __init__(self, **kwargs):
        super(TextCNN, self).__init__()
        # Update Default Parameters
        config = dict()
        for k, v in TextCNN.DEFAULT_CONFIG.items():
            config[k] = kwargs[k] if k in kwargs else v
        # Hyperparameters
        self.model = config['model']
        self.max_sent_len = config['max_sent_len']
        self.word_dim = config['word_dim']
        self.vocab_size = config['vocab_size']
        self.class_size = config['class_size']
        self.filters = config['filters']
        self.filter_num = config['filter_num']
        self.dropout_prob = config['dropout_prob']
        self.wv_matrix = config['wv_matrix']
        self.return_layers = config['return_layers'] if 'return_layers' in kwargs else ['fc']
        self.in_channel = 1
        # Validate Hyperparameters
        assert (len(self.filters) == len(self.filter_num))
        if self.model != 'rand':
            self.vocab_size = self.wv_matrix.shape[0] - 2
            self.word_dim = self.wv_matrix.shape[1]
        # one for UNK and one for zero padding
        self.embedding = nn.Embedding(self.vocab_size + 2, self.word_dim, padding_idx=self.vocab_size + 1)
        if self.model != 'rand':
            self.embedding.weight.data.copy_(torch.from_numpy(self.wv_matrix))
            if self.model == 'static':
                self.embedding.weight.requires_grad = False
            elif self.model == 'multichannel':
                self.embedding2 = nn.Embedding(self.vocab_size + 2, self.word_dim, padding_idx=self.vocab_size + 1)
                self.embedding2.weight.data.copy_(torch.from_numpy(self.wv_matrix))
                self.embedding2.weight.requires_grad = False
                self.in_channel = 2
        for i in range(len(self.filters)):
            temp = nn.Conv1d(self.in_channel, self.filter_num[i], self.word_dim * self.filters[i], stride=self.word_dim)
            setattr(self, 'conv_%i' % i, temp)
        self.fc = nn.Linear(sum(self.filter_num), self.class_size)

    def forward(self, text):
        outputs = {}
        x = self.embedding(text).view(-1, 1, self.word_dim * self.max_sent_len)
        outputs['embedding'] = x
        if self.model == 'multichannel':
            x2 = self.embedding2(text).view(-1, 1, self.word_dim * self.max_sent_len)
            outputs['embedding2'] = x2
            x = torch.cat((x, x2), 1)
        conv_results = []
        for i in range(len(self.filters)):
            conv_result = F.relu(getattr(self, 'conv_%i' % i)(x))
            kernel_size = self.max_sent_len - self.filters[i] + 1
            conv_result = F.max_pool1d(conv_result, kernel_size).view(-1, self.filter_num[i])
            outputs['conv_%i' % i] = conv_result
            conv_results.append(conv_result)
        x = torch.cat(conv_results, 1)
        x = F.dropout(x, p=self.dropout_prob, training=self.training)
        x = self.fc(x)
        outputs['fc'] = x
        return torch.cat([outputs[layer] for layer in self.return_layers], 1)


class AttentionNet(nn.Module):
    def __init__(self, **kwargs):
        super(AttentionNet, self).__init__()
        self.batch_size = kwargs['batch_size']
        self.wv_matrix = kwargs['wv_matrix']
        # self.vocab_size = kwargs['vocab_size']
        # self.word_dim = kwargs['word_dim']
        self.vocab_size = self.wv_matrix.shape[0]
        self.word_dim = self.wv_matrix.shape[1]
        self.hidden_size = kwargs['hidden_size']
        self.output_size = kwargs['output_size']
        # Layer Configs
        self.embedding = nn.Embedding(self.vocab_size, self.word_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(self.wv_matrix))
        self.embedding.weight.requires_grad = True
        self.lstm = nn.LSTM(self.word_dim, self.hidden_size, dropout=0.5)
        self.dropout = nn.Dropout(0.5)
        self.label = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, text):
        batch_size, _ = text.shape
        x = self.embedding(text)
        x = x.permute(1, 0, 2)
        h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        # output.size() = (seq_len, batch_size, hidden_size)
        # h_n.size() = (1, batch_size, hidden_size)
        output = output.permute(1, 0, 2)
        # output.size() = (batch_size, seq_len, hidden_size)
        attn_output = AttentionNet.attention(output, h_n)
        # attn_output.size() = (batch_size, hidden_size)
        attn_output = self.dropout(attn_output)
        logits = self.label(attn_output)
        return logits

    @classmethod
    def attention(cls, hidden_state, final_state):
        """Self attention on 
        
        :param hidden_state:
            Tensor of shape (batch_size, seq_len, hidden_size)
             containing the output features (h_t) from the last layer of the LSTM, for each t.
        :param final_state:
            Tensor of shape (1, batch_size, hidden_size)
             containing the hidden state for t = seq_len.
        :return: 
        """
        attn_weights = torch.bmm(hidden_state, final_state.squeeze(0).unsqueeze(2)).squeeze(2)
        # attn_weights.shape() = (batch_size, seq_len)
        soft_attn_weights = F.softmax(attn_weights, 1)
        hidden_state = torch.bmm(hidden_state.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return hidden_state
