from utils import showPlot, asMinutes, timeSince, showAttention
from torch import optim
from model import EncoderRNN, AttnDecoderRNN, DecoderRNN
import cloudpickle
from data import SOS_token, EOS_token, normalizeString
import torch
import torch.nn as nn
import time
import random
import os
import code


# Inform the user if the notebook uses GPU or CPU.
def set_device():
    """
    Set the device. CUDA if available, CPU otherwise

    Args:
    None

    Returns:
    Nothing
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "cpu"
    else:
        device = "cpu"

    print("Using {} device".format(device))

    return device


device = set_device()


class EncoderDecoder(object):
    """EncoderDecoder"""

    def __init__(self, hidden_size=128, input_vocab_len=10000, output_vocab_len=10000, dropout_p=0.1,
                 teacher_forcing_ratio=0.5, max_length=10, learning_rate=0.01, simple=False,
                 bidirectional=False, dot=False, bilinear=False):
        super(EncoderDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_vocab_len = input_vocab_len
        self.output_vocab_len = output_vocab_len
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.simple = simple
        self.dot = dot
        self.bilinear = bilinear
        self.bidirectional = bidirectional
        self.teacher_forcing_ratio = teacher_forcing_ratio

        if self.bidirectional:
            self.encoder = code.define_bi_encoder(
                input_vocab_len, hidden_size).to(device)

        else:
            self.encoder = EncoderRNN(input_vocab_len, hidden_size).to(device)

        if self.simple:

            self.decoder = code.define_simple_decoder(
                hidden_size, input_vocab_len, output_vocab_len, max_length).to(device)
        else:
            if self.dot:
                self.decoder = code.AttnDecoderRNNDot(
                    hidden_size, output_vocab_len, dropout_p=dropout_p, max_length=max_length).to(device)
            elif self.bilinear:
                self.decoder = code.AttnDecoderRNNBilinear(
                    hidden_size, output_vocab_len, dropout_p=dropout_p, max_length=self.max_length).to(device)
            else:
                self.decoder = AttnDecoderRNN(
                    hidden_size, output_vocab_len, dropout_p=dropout_p, max_length=self.max_length).to(device)

        self.encoder_optimizer = None
        self.decoder_optimizer = None
        self.criterion = None
        self.input_lang = None
        self.output_lang = None

    def indexesFromSentence(self, lang, sentence):
        return [lang.word2index[word] for word in sentence.split(' ')]

    def tensorFromSentence(self, lang, sentence):
        indexes = self.indexesFromSentence(lang, sentence)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

    def tensorsFromPair(self, pair):
        input_tensor = self.tensorFromSentence(self.input_lang, pair[0])
        target_tensor = self.tensorFromSentence(self.output_lang, pair[1])
        return (input_tensor, target_tensor)

    def train(self, input_tensor, target_tensor):
        encoder_hidden = self.encoder.initHidden()

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(
            self.max_length, self.hidden_size, device=device)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei], encoder_hidden)

            if self.bidirectional:
                encoder_output = code.fix_bi_encoder_output_dim(
                    encoder_output, self.hidden_size)

            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)

        if self.bidirectional:
            decoder_hidden = code.fix_bi_encoder_hidden_dim(encoder_hidden)
        else:
            decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random(
        ) < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                if self.simple:
                    decoder_output, decoder_hidden = code.run_simple_decoder(self.decoder, decoder_input,
                                                                             encoder_hidden, decoder_hidden, encoder_outputs)
                else:
                    decoder_output, decoder_hidden, decoder_attention = self.decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                loss += self.criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                if self.simple:
                    decoder_output, decoder_hidden = code.run_simple_decoder(self.decoder, decoder_input,
                                                                             encoder_hidden, decoder_hidden, encoder_outputs)
                else:
                    decoder_output, decoder_hidden, decoder_attention = self.decoder(
                        decoder_input, decoder_hidden, encoder_outputs)

                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += self.criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == EOS_token:
                    break

        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item() / target_length

    def trainIters(self, pairs, input_lang, output_lang, n_iters, print_every=1000, plot_every=100):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        self.input_lang = input_lang
        self.output_lang = output_lang
        self.encoder_optimizer = optim.SGD(
            self.encoder.parameters(), lr=self.learning_rate)
        self.decoder_optimizer = optim.SGD(
            self.decoder.parameters(), lr=self.learning_rate)
        selected_pairs = [random.choice(pairs) for i in range(n_iters)]
        training_pairs = [self.tensorsFromPair(
            pair) for pair in selected_pairs]
        self.criterion = nn.NLLLoss()

        for iter in range(1, n_iters + 1):
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]
            loss = self.train(input_tensor, target_tensor)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                             iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        showPlot(plot_losses)

        return print_loss_avg

    def evaluate(self, sentence):
        with torch.no_grad():
            input_tensor = self.tensorFromSentence(self.input_lang, sentence)
            input_length = input_tensor.size()[0]
            encoder_hidden = self.encoder.initHidden()

            encoder_outputs = torch.zeros(
                self.max_length, self.encoder.hidden_size, device=device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei],
                                                              encoder_hidden)
                if self.bidirectional:
                    encoder_output = code.fix_bi_encoder_output_dim(
                        encoder_output, self.hidden_size)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

            if self.bidirectional:
                decoder_hidden = code.fix_bi_encoder_hidden_dim(encoder_hidden)
            else:
                decoder_hidden = encoder_hidden

            decoded_words = []
            if not self.simple:
                decoder_attentions = torch.zeros(
                    self.max_length, self.max_length)

            for di in range(self.max_length):
                if self.simple:
                    decoder_output, decoder_hidden = code.run_simple_decoder(self.decoder, decoder_input,
                                                                             encoder_hidden, decoder_hidden, encoder_outputs)
                else:
                    decoder_output, decoder_hidden, decoder_attention = self.decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    decoder_attentions[di] = decoder_attention.data

                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(
                        self.output_lang.index2word[topi.item()])

                decoder_input = topi.squeeze().detach()

        if not self.simple:
            return decoded_words, decoder_attentions[:di + 1]
        else:
            return decoded_words, None

    @classmethod
    def load(cls, directory):
        with open(os.path.join(directory, 'args.pkl'), 'rb') as f:
            params = cloudpickle.load(f)

        model = EncoderDecoder(params['hidden_size'], params['input_vocab_len'], params['output_vocab_len'],
                               dropout_p=params['dropout_p'], teacher_forcing_ratio=params['teacher_forcing_ratio'],
                               max_length=params['max_length'], learning_rate=params['learning_rate'], simple=params['simple'], bidirectional=params['bidirectional'], dot=params['dot'], bilinear=params['bilinear'])

        model.input_lang = params['input_lang']
        model.output_lang = params['output_lang']
        model.encoder.load_state_dict(torch.load(
            os.path.join(directory, 'encoder.pt'), map_location=lambda storage, loc: storage
        ).state_dict())
        model.decoder.load_state_dict(torch.load(
            os.path.join(directory, 'decoder.pt'), map_location=lambda storage, loc: storage
        ).state_dict())

        return model

    def save(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

        def create_save_model(model, path):
            return torch.save(model, path)

        create_save_model(self.encoder, directory + 'encoder.pt')
        create_save_model(self.decoder, directory + 'decoder.pt')

        with open(os.path.join(directory, 'args.pkl'), 'wb') as f:
            cloudpickle.dump({
                'input_lang': self.input_lang,
                'output_lang': self.output_lang,
                'dropout_p': self.dropout_p,
                'teacher_forcing_ratio': self.teacher_forcing_ratio,
                'max_length': self.max_length,
                'learning_rate': self.learning_rate,
                'hidden_size': self.hidden_size,
                'input_vocab_len': self.input_vocab_len,
                'output_vocab_len': self.output_vocab_len,
                'simple': self.simple,
                'bidirectional': self.bidirectional,
                'dot': self.dot,
                'bilinear': self.bilinear,
            }, f)

    def evaluatePairs(self, pairs, rand=True, n=10, plot=False):
        n = n if rand else len(pairs)
        outputs = []
        for i in range(n):
            if rand:
                pair = random.choice(pairs)
            else:
                pair = pairs[i]
            print('>', pair[0])
            print('=', pair[1])
            output_words, attentions = self.evaluate(pair[0])
            if plot and not self.simple:
                plt.matshow(attentions.numpy())
            output_sentence = ' '.join(output_words[:-1])
            outputs.append((output_sentence, pair[1]))
            print('<', output_sentence)
            print('')
        return outputs

    def evaluateAndShowAttention(self, input_sentence):
        output_words, attentions = self.evaluate(
            normalizeString(input_sentence))
        print('input =', input_sentence)
        print('output =', ' '.join(output_words))
        if not self.simple:
            showAttention(normalizeString(input_sentence),
                          output_words, attentions[:, :len(output_words)])
        else:
            print("Not an attention based model as per the parameter 'simple' !")
