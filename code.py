import model
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import DecoderRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def define_simple_decoder(hidden_size, input_vocab_len, output_vocab_len, max_length):
    """ Provides a simple decoder instance
    
    :param hidden_size:
    :param input_vocab_len
    :param output_vocab_len
    :param max_length
    :return: a simple decoder instance
    """
    decoder = None
    decoder = DecoderRNN(hidden_size, output_vocab_len)
    return decoder


def run_simple_decoder(simple_decoder, decoder_input, encoder_hidden, decoder_hidden, encoder_outputs):
    """ Runs the simple_decoder

    :param simple_decoder: the simple decoder object
    :param decoder_input:
    :param decoder_hidden:
    :param encoder_hidden:
    :param encoder_outputs:
    :return: The appropriate values
    """
    results = None
    output, hidden = simple_decoder(decoder_input, decoder_hidden)
    return output, hidden 


class BidirectionalEncoderRNN(nn.Module):
    """class definition for BidirectionalEncoderRNN
    """

    def __init__(self, input_size, hidden_size):
        """
        :param input_size:
        :param hidden_size:
        """
        super(BidirectionalEncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.bi_gru = nn.GRU(hidden_size, hidden_size, bidirectional=True)


    def forward(self, input, hidden):
        """
        :param input:
        :param hidden:
        :return: output, hidden
        """
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        bi_output, bi_hidden = self.bi_gru(output, hidden)
        return bi_output, bi_hidden


    def initHidden(self):
        return torch.zeros(1*2, 1, self.hidden_size, device=device)


def define_bi_encoder(input_vocab_len, hidden_size):
    """ Defines bidirectional encoder RNN

    :param input_vocab_len:
    :param hidden_size:
    :return: bidirectional encoder RNN
    """
    encoder = None
    encoder = BidirectionalEncoderRNN(input_vocab_len, hidden_size)
    return encoder


def fix_bi_encoder_output_dim(encoder_output, hidden_size):
    """
    :param encoder_output:
    :param hidden_size:
    :return: output
    """
    output = None
    bi_out, rev_out = torch.split(encoder_output, hidden_size, 2)
    output = torch.add(bi_out, rev_out)
    return output


def fix_bi_encoder_hidden_dim(encoder_hidden):
    """
    :param encoder_hidden:
    :return: output
    """

    output = None
    bi_hidden, rev_hidden = torch.split(encoder_hidden, 1)
    output = bi_hidden
    return output


class AttnDecoderRNNDot(nn.Module):
    """
    class definition for AttnDecoderRNNDot
    """

    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=10):
        super(AttnDecoderRNNDot, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
       

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = torch.softmax(torch.matmul(
            hidden[0], encoder_outputs.T), dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights
        

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNNBilinear(nn.Module):
    """
    class definition for AttnDecoderRNNBilinear
    """

    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=10):
        super(AttnDecoderRNNBilinear, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = torch.softmax(torch.matmul(
            self.attn(hidden[0]), encoder_outputs.T), dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights


    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
