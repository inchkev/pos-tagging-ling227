"""
viterbi.py

Viterbi POS tagger implementation in Python

LING 227 Final Project
Kevin Chen, Nick Schoelkopf, Neal Ma
"""

import math
from collections import defaultdict

import nltk
import pyconll
import pyconll.util

UD_ENGLISH_BASE = './UD_English-EWT/en_ewt-ud-'
UD_ENGLISH_TRAIN = UD_ENGLISH_BASE + 'train.conllu'
UD_ENGLISH_DEV =   UD_ENGLISH_BASE + 'dev.conllu'
UD_ENGLISH_TEST =  UD_ENGLISH_BASE + 'test.conllu'
UNKNOWN_IDENTIFIER = 'UNKNOWN_IDENTIFIER'
RARE_COUNT = 1


def viterbi(o, a, b, pi, states):
    viterbi = defaultdict(float)
    backpointer = defaultdict(float)
    T = len(o)

    # initialization step
    for s in states:
        viterbi[(s, 1)] = pi[s] + b[s].logprob(o[0])
        backpointer[(s, 1)] = 0

    # recursion step
    for t in range(2, T+1):
        for s in states:
            prev = [(
                viterbi[(s_, t-1)] + a[s_].logprob(s) + b[s].logprob(o[t-1]), s_)
                for s_ in states]

            viterbi[(s, t)], backpointer[(s, t)] = max(prev, key=lambda pair: pair[0])

    # termination step
    last = [(viterbi[(s, T)], s) for s in states]
    bestpathlogprob, bestpathpointer = max(last, key=lambda pair: pair[0])

    # follow backpointer starting at bestpathpointer
    bestpath = [bestpathpointer]
    for i in range(T, 1, -1):
        bestpath.insert(0, backpointer[(bestpath[0], i)])

    return bestpath, bestpathlogprob


if __name__ == '__main__':

    training_set = pyconll.iter_from_file(UD_ENGLISH_TRAIN)

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
    word_tags = [(pos, form if (form == 'START' or form == 'END' or training_words[form] > RARE_COUNT) else UNKNOWN_IDENTIFIER) for pos, form in word_tags]

    tags = [tag for tag, _ in word_tags]

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

    dev_set = pyconll.iter_from_file(UD_ENGLISH_DEV)
    correct_words = 0
    total_words = 0

    for sentence in dev_set:
        words = [token.form if (training_words[token.form] > RARE_COUNT) else UNKNOWN_IDENTIFIER for token in sentence]
        words = ['START'] + words + ['END']
        correct_pos = [token.upos for token in sentence]

        estimated_pos, prob = viterbi(words, A, B, pi, states)
        for i, pos in enumerate(estimated_pos[1:-1]):
            total_words += 1
            if pos == correct_pos[i]:
                correct_words += 1

    print(f'Accuracy: {(100 * float(correct_words) / total_words):.5f}')
    # should be around 87.93
