from __future__ import unicode_literals, print_function, division
import argparse
from data import readFile, prepareData
from utils import writeToFile, compute_bleu
from seq2seq import EncoderDecoder
import time
import random
import numpy as np
import torch
import os


def set_seed(seed=None, seed_torch=True):
    """
    Function that controls randomness. NumPy and random modules must be imported.

    Args:
        seed : Integer
        A non-negative integer that defines the random state. Default is `None`.
        seed_torch : Boolean
        If `True` sets the random seed for pytorch tensors, so pytorch module
        must be imported. Default is `True`.

    Returns:
        Nothing.
    """
    if seed is None:
        seed = np.random.choice(2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print(f'Random seed {seed} has been set.')


parser = argparse.ArgumentParser(description='Neural Machine Translation')
parser.add_argument('--train-file', type=str, default="eng-fra.train.small.txt",
                    help='input file for training (default: eng-fra.train.small.txt)')
parser.add_argument('--test-file', type=str, default="eng-fra.test.small.txt",
                    help='input file for evaluation (default: eng-fra.test.small.txt)')
parser.add_argument('--output-dir', type=str, default="results/",
                    help='output directory to save the model(default: results/)')
parser.add_argument('--max-length', type=int, default=10,
                    help='Maximum sequence length (default: 10)')
parser.add_argument('--tfr', type=float, default=0.5,
                    help='Teacher Forcing Ratio (default: 0.5)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning Rate (default: 0.01)')
parser.add_argument('--drop', type=float, default=0.1,
                    help='Dropout Probability (default: 0.1)')
parser.add_argument('--hidden-size', type=int, default=126,
                    help='Size of hidden layer (default: 126)')
parser.add_argument('--n-iters', type=int, default=10000,
                    help='Number of Iterations (default: 10000)')
parser.add_argument('--plot-every', type=int, default=100,
                    help='Plot after (default: 100)')
parser.add_argument('--print-every', type=int, default=1000,
                    help='Print after(default: 1000)')
parser.add_argument('--eval', default=False, action='store_true',
                    help='Run the model on test file')
parser.add_argument('--simple', default=False, action='store_true',
                    help='Run the simple decoder')
parser.add_argument('--bidirectional', default=False, action='store_true',
                    help='Run the bidirectional encoder')
parser.add_argument('--dot', default=False, action='store_true',
                    help='Run the Attention decoder with dot type')
parser.add_argument('--bilinear', default=False, action='store_true',
                    help='Run the Attention decoder with bilinear type')
# parser.add_argument('--device', type=str, default="cpu",
#                     help='Choose device within "cpu", "cuda" or "mps"')


def main():

    global args, max_length
    args = parser.parse_args()

    if args.eval:

        if not os.path.exists(args.output_dir):
            print("Output directory do not exists")
            exit(0)
        try:
            model = EncoderDecoder().load(args.output_dir)
            print("Model loaded successfully")
        except:
            print("The trained model could not be loaded...")
            exit()

        test_pairs = readFile(args.test_file)
        outputs = model.evaluatePairs(test_pairs,  rand=False)
        writeToFile(outputs, os.path.join(args.output_dir, "output.pkl"))
        reference = []
        hypothesis = []
        for (hyp, ref) in outputs:
            reference.append([ref.split(" ")])
            hypothesis.append(hyp.split(" "))

        bleu_score = compute_bleu(reference, hypothesis)
        print("Bleu Score: " + str(bleu_score))

        # print(model.evaluateAndShowAttention(
        #     "L'anglais n'est pas facile pour nous."))
        # print(model.evaluateAndShowAttention(
        #     "J'ai dit que l'anglais est facile."))
        # print(model.evaluateAndShowAttention(
        #     "Je fais un blocage sur l'anglais."))
        # print(model.evaluateAndShowAttention(
        #     "Je n'ai pas dit que l'anglais est une langue facile."))

    else:
        input_lang, output_lang, pairs = prepareData(args.train_file)
        print(random.choice(pairs))
        model = EncoderDecoder(args.hidden_size, input_lang.n_words, output_lang.n_words, args.drop,
                               args.tfr, args.max_length, args.lr, args.simple, args.bidirectional, args.dot, args.bilinear)
        loss = model.trainIters(pairs, input_lang, output_lang, args.n_iters,
                                print_every=args.print_every, plot_every=args.plot_every)
        model.evaluatePairs(pairs)
        model.save(args.output_dir)


if __name__ == '__main__':

    SEED = 34
    set_seed(seed=SEED, seed_torch=True)
    # DEVICE = set_device()
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
