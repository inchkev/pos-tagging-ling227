#ifndef __PARSE_HH__
#define __PARSE_HH__

#include <bits/stdc++.h>
using namespace std;

/**
 * Trains the model from a given file.
 */
void train(FILE *in, int *start, int *end, int *tags, int **transition);

/**
 * Adds to the HMM from a single sentence in a given file.
 */
int train_sentence(FILE *in, int *start, int *end, int *tags, int **transition);

/**
 * Evaluates the trained HMM using a Viterbi algorithm and returns a double 
 * representing the proportion of the testing data the model correctly labeled.
 */
double eval_viterbi(FILE *in, int *start, int *end, int *tags, int **transition);

/**
 * Reads in a sentence from the given file stream, returns an array of strings
 * with even strings being the form of the word and odd strings being the assigned
 * part of speech tag of the word
 */
char **get_sentence(FILE *in, int *len);

/**
 * Evaluates a given sentence from a trained HMM using the Viterbi algorithm. Increments
 * "total" for each word in the sentence and increments "success" for each correct tag.
 */
void eval_sent(int *start, int *end, int *tags, int**transition, char **sentence, int len, size_t *success, size_t *total);

/**
 * Splits a line and returns a string array where the first string is the form, second string
 * is the lemma, and third string is the Universal Part of Speech Tag for the given word.
 */
char **split(char *line, int length);

/**
 * Reads and returns a single string from a given file stream.
 */
char *get_line(FILE *in, int *len);

/**
 * Returns the tag number associated with the given standard part of speech tag passed in.
 */
int find_tag(char *tag);

/**
 * Counts all words with occurances under REMOVE_THRESH and counts each as a signle "unknown" identifier.
 */
void filter();

/**
 * Number of universal part of speech tags.
 */
const int POS_NUM = 17;

/**
 * Threshold mentioned in filter().
 */
const int REMOVE_THRESH = 1;

/**
 * Unknown identifier char *.
 */
char unknown[2] = {'U', 'K'};

/**
 * Set of standard part of speech tags used in find_tag().
 */
const char *POS_STANDARD[] = {"adj", "adp", "adv", "aux", "cconj", "det", "intj", "noun", "num", "part", "pron", "propn", "punct", "sconj", "sym", "verb", "x"};

#endif
