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

UD_ENGLISH_BASE = './UD_English-EWT/en_ewt-ud-'
UD_ENGLISH_TRAIN = UD_ENGLISH_BASE + 'train.conllu'
UD_ENGLISH_DEV =   UD_ENGLISH_BASE + 'dev.conllu'
UD_ENGLISH_TEST =  UD_ENGLISH_BASE + 'test.conllu'
UNKNOWN_IDENTIFIER = 'UNKNOWN_IDENTIFIER'
RARE_COUNT = 1


def viterbi(o, a, b, pi, states):
    viterbi = defaultdict(float)
    backpointer = defaultdict(int)
    T = len(o)

    # initialization step
    for s in states:
        viterbi[(s, 1)] = pi[s] + b[s].logprob(o[0])
        # backpointer[(s, 1)] = 0

    # recursion step
    for t in range(2, T+1):
        for s in states:
            prev = (
                (viterbi[(s_, t-1)] + a[s_].logprob(s) + b[s].logprob(o[t-1]), s_)
                for s_ in states)

            viterbi[(s, t)], backpointer[(s, t)] = max(prev, key=lambda pair: pair[0])

    # termination step
    last = ((viterbi[(s, T)], s) for s in states)
    bestpathlogprob, bestpathpointer = max(last, key=lambda pair: pair[0])

    # follow backpointer starting at bestpathpointer
    bestpath = [bestpathpointer]
    for i in range(T, 1, -1):
        bestpath.append(backpointer[(bestpath[T - i], i)])

    return tuple(reversed(bestpath)), bestpathlogprob


if __name__ == '__main__':
    start = time.time()

    training_set = iter_from_file(UD_ENGLISH_TRAIN)

    states = {'START', 'END'}
    training_words = defaultdict(int)
    word_tags = []
    for sentence in training_set:
        # add start tag
        word_tags.append(('START', 'START'))

        # add (tag, word) tuples
        for token in sentence:
            word_tags.append((token.upos, token.form))
            states.add(token.upos)
            training_words[token.form] += 1

        # add end tag
        word_tags.append(('END', 'END'))

    # set identifiers which appear RARE_COUNT times or less to UNKNOWN_IDENTIFIER
    word_tags = tuple((pos, form if (form == 'START' or form == 'END' or training_words[form] > RARE_COUNT) else UNKNOWN_IDENTIFIER) for pos, form in word_tags)

    tags = tuple(tag for tag, _ in word_tags)

    # get transition matrix, P(ti | ti-1)
    cfdist_tags = nltk.ConditionalFreqDist(nltk.bigrams(tags))
    A = nltk.ConditionalProbDist(cfdist_tags, nltk.MLEProbDist)

    # get emission matrix, P(wi | ti)
    cfdist_words = nltk.ConditionalFreqDist(word_tags)
    B = nltk.ConditionalProbDist(cfdist_words, nltk.MLEProbDist)

    # get π
    pi = defaultdict(float)
    pi['START'] = 1.0

    # --- testing on dev set ---
    print('Done with training.')
    start_dev = time.time()

    dev_set = iter_from_file(UD_ENGLISH_DEV)
    correct_words = 0
    total_words = 0

    for sentence in dev_set:
        words = ('START', *(token.form if (training_words[token.form] > RARE_COUNT) else UNKNOWN_IDENTIFIER for token in sentence), 'END')
        correct_pos = (token.upos for token in sentence)

        estimated_pos, _ = viterbi(words, A, B, pi, states)

        # compare estimated and correct tags
        for pair in zip(estimated_pos[1:-1], correct_pos):
            total_words += 1
            if pair[0] == pair[1]:
                correct_words += 1

    print(f'Dev Accuracy: {(100 * float(correct_words) / total_words):.5f}')
    # should be 87.93469 for English

    # runtime stats
    end = time.time()
    print()
    print(f'Total: {end - start:.2f} seconds')
    print(f'Dev:   {end - start_dev:.2f} seconds')
