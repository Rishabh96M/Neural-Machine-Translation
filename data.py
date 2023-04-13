from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re

# import random
# import string

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS and UNK

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readFile(file_name, reverse=True):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open('data/%s' % (file_name),
                 encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in line.split('\t')] for line in lines]
    print("Read %s sentence pairs" % len(pairs))

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]

    return pairs


def prepareData(file_name, lang1="eng", lang2="fra", reverse=True):

    pairs = readFile(file_name, reverse=reverse)

    # Reverse pairs, make Lang instances
    if reverse:
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
        # print(pair)
        # print(len(pair))
        # break
    print("Counted words:")
    input_count = 0
    for word in input_lang.word2count:
        input_count += input_lang.word2count[word]

    output_count = 0
    for word in output_lang.word2count:
        output_count += output_lang.word2count[word]

    print(input_lang.name, input_lang.n_words,
          input_count, round(input_count/len(pairs), 3))
    print(output_lang.name, output_lang.n_words,
          output_count, round(output_count/len(pairs), 3))

    return input_lang, output_lang, pairs
