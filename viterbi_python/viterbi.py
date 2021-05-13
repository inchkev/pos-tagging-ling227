"""
viterbi.py

Viterbi POS tagger implementation in Python

LING 227 Final Project
Kevin Chen, Nick Schoelkopf, Neal Ma
"""

import math
import time
from collections import defaultdict

import nltk
from pyconll import iter_from_file

UD_BASE  = './UD_English-EWT/en_ewt-ud-'
# UD_BASE  = './UD_Chinese-GSDSimp/zh_gsdsimp-ud-'
# UD_BASE  = './UD_Korean-Kaist/ko_kaist-ud-'
# UD_BASE  = './UD_German-GSD/de_gsd-ud-'
# UD_BASE  = './UD_Spanish-GSD/es_gsd-ud-'
# UD_BASE  = './UD_Indonesian-GSD/id_gsd-ud-'
# UD_BASE  = './UD_Dutch-Alpino/nl_alpino-ud-'
UD_TRAIN = UD_BASE + 'train.conllu'
UD_DEV   = UD_BASE + 'dev.conllu'
UD_TEST  = UD_BASE + 'test.conllu'
UNKNOWN_IDENTIFIER = 'UNKNOWN_IDENTIFIER'
RARE_COUNT = 1


def viterbi(o, a, b, pi, states):
    """Run the viterbi algorithm

    Args:
        o: A list or tuple representing the string.
        a: A ConditionalProbDist representing the transition matrix A.
        b: A ConditionalProbDist representing the emission matrix B.
        pi: A defaultdict representing the initial state distribution π.
        states: A set or dictionary representing all possible states (POS tags).

    Returns:
        bestpath: A tuple representing the tagged POS sequence.
        bestpathlogprob: A float of the log probability of the best path.
    """
    viterbi = defaultdict(float)
    backpointer = defaultdict(int)
    T = len(o)

    # initialization step
    for s in states:
        viterbi[(s, 1)] = pi[s] + b[s].logprob(o[0])

    # recursion step
    for t in range(2, T+1):
        for s in states:
            prev = (
                (viterbi[(s_, t-1)] + a[s_].logprob(s) + b[s].logprob(o[t-1]), s_)
                for s_ in states)

            # woo dynamic programming!
            viterbi[(s, t)], backpointer[(s, t)] = max(prev, key=lambda pair: pair[0])

    # termination step
    last = ((viterbi[(s, T)], s) for s in states)
    bestpathlogprob, bestpathpointer = max(last, key=lambda pair: pair[0])

    # follow backpointer starting at bestpathpointer
    bestpath = [bestpathpointer]
    for i in range(T, 1, -1):
        bestpath.append(backpointer[(bestpath[T - i], i)])

    # reverse best path and return its log probability
    return tuple(reversed(bestpath)), bestpathlogprob


if __name__ == '__main__':
    start = time.time()

    training_set = iter_from_file(UD_TRAIN)

    states = {'START', 'END'}           # set of all unique states
    training_words = defaultdict(int)   # count word occurrences in training set
    word_tags = []
    for sentence in training_set:
        # add start tag
        word_tags.append(('START', 'START'))

        # add (tag, word) tuples
        for token in sentence:
            # ignore if upos is None
            if token.upos is None:
                continue
            word_tags.append((token.upos, token.form))
            states.add(token.upos)
            training_words[token.form] += 1

        # add end tag
        word_tags.append(('END', 'END'))

    # set identifiers which appear <= RARE_COUNT to UNKNOWN_IDENTIFIER
    word_tags = tuple((pos, form if (form == 'START' or
                                     form == 'END' or
                                     training_words[form] > RARE_COUNT)
                                 else UNKNOWN_IDENTIFIER)
        for pos, form in word_tags)

    # extract only tags from word_tags
    tags = tuple(tag for tag, _ in word_tags)

    # get transition matrix, P(ti | ti-1)
    cfdist_tags = nltk.ConditionalFreqDist(nltk.bigrams(tags))
    A = nltk.ConditionalProbDist(cfdist_tags, nltk.MLEProbDist)

    # get emission matrix, P(wi | ti)
    cfdist_words = nltk.ConditionalFreqDist(word_tags)
    B = nltk.ConditionalProbDist(cfdist_words, nltk.MLEProbDist)

    print('Done with training.')

    # get initial state distribution π
    pi = defaultdict(float)
    pi['START'] = 1.0

    #
    # --- classify ---
    #

    dev_set = iter_from_file(UD_DEV)
    # dev_set = iter_from_file(UD_TEST)

    start_dev = time.time()
    correct_words = 0
    total_words = 0

    for sentence in dev_set:
        # filter if upos is None
        new_sentence = tuple(filter(lambda x: x.upos is not None, sentence))

        words = (
            'START',
            *(token.form if (training_words[token.form] > RARE_COUNT)
                         else UNKNOWN_IDENTIFIER for token in new_sentence),
            'END')
        correct_pos = [token.upos for token in new_sentence]

        # tag sentence with viterbi algorithm
        estimated_pos, _ = viterbi(words, A, B, pi, states)

        # compare estimated and correct tags
        for pair in zip(estimated_pos[1:-1], correct_pos):
            total_words += 1
            if pair[0] == pair[1]:
                correct_words += 1

    # print dev set accuracy
    print(f'Accuracy: {(100 * float(correct_words) / total_words):.5f}')
    # should be 88.21 for English

    # print runtime stats
    end = time.time()
    print()
    print(f'Total:    {end - start:.2f} seconds')
    print(f'Classify: {end - start_dev:.2f} seconds')
