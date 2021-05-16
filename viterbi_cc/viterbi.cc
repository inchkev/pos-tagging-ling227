#include <cstdio>
#include <stdio.h>
#include <cstring>
#include <cstdlib>
#include <cinttypes>
#include <cassert>
#include <stdbool.h>
#include <stdlib.h>
#include <cmath>
#include <float.h>

#include <chrono>
#include <sys/time.h>
#include <ctime>

#include "viterbi.hh"

#include <bits/stdc++.h>
using namespace std;

/**
 * viterbi.cc
 * 
 * Viterbi POS Tagger implementation in C++
 * 
 * LING 227 Final Project
 * Kevin Chen, Nick Schoelkopf, Neal Ma
 * 
 * Comments explaining each function are found in the header file (viterbi.hh)
 * Input files must be in standard .conllu format
 */

typedef struct _word
{
    int count;
    int pos[POS_NUM + 1];
} word;

unordered_map<string, word *> emission;

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printf("USAGE: ./Viterbi [training_file_path] [testing_file_path]\n");
        return 1;
    }
    int *start = (int *) malloc(sizeof(int) * (POS_NUM + 1));
    int *end = (int *) malloc(sizeof(int) * (POS_NUM + 1));
    int *tags = (int *) malloc(sizeof(int) * (POS_NUM + 1));
    int **transition = (int **) malloc(sizeof(int *) * (POS_NUM + 1));
    for (int i = 0; i < (POS_NUM + 1); i++)
    {
        transition[i] = (int *) malloc(sizeof(int) * (POS_NUM + 1));
        start[i] = 0; end[i] = 0;
        for (int j = 0; j < (POS_NUM + 1); j++)
        {
            transition[i][j] = 0;

        }
    }
    FILE *f = fopen(argv[1], "r");
    train(f, start, end, tags, transition);
    fclose(f);
    filter();
    FILE *f2 = fopen(argv[2], "r");
    auto start_t = std::chrono::steady_clock::now();
    double correct = eval_viterbi(f2, start, end, tags, transition);
    auto end_t = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_t - start_t;
    printf("EVAL TIME: %lf\n",(elapsed_seconds.count()));
    printf("SUCCESS: %lf\n", correct * 100.0);
    fclose(f2);
}

double eval_viterbi(FILE *in, int *start, int *end, int *tags, int **transition)
{
    size_t success = 0;
    size_t total = 0;
    int len = -1;
    int count = 0;
    char **sentence;
    while ((sentence = get_sentence(in, &len)) != NULL)
    {
        if (len == 0)
        {
            break;
        }
        // dumb_eval(start, end, tags, transition, sentence, len, &success, &total);
        eval_sent(start, end, tags, transition, sentence, len, &success, &total);
    }
    printf("SUCCESSFULLY TAGGED: %lu\nTOTAL TAGS: %lu\n", success, total);
    return ((double) success) / (total);
}

void dumb_eval(int *start, int *end, int *tags, int**transition, char **sentence, int len, size_t *success, size_t *total)
{
    int k = 0;
    int max_unknown = -1;
    for (int i = 0; i < len; i++)
    {
        if (emission[unknown]->pos[i] > emission[unknown]->pos[k])
        {
            k = i;
        }
    }
    for (int i = 0; i < len; i++)
    {
        double max_prob = -1.0 * DBL_MAX;
        int max_pos = 0;
        int a = i * 2;
        int b = i * 2 + 1;
        for (int j = 0; j < POS_NUM + 1; j++)
        {
            double em_count = 0;
            if (emission.find(sentence[a]) == emission.end() || emission[sentence[a]]->count <= REMOVE_THRESH)
            {
                em_count = ((double)emission[unknown]->pos[k]) / ((double) emission[unknown]->count);
            }
            else
            {
                em_count = ((double)emission[sentence[a]]->pos[j]) / ((double) emission[sentence[a]]->count);
            }
            if (em_count > max_prob)
            {
                max_prob = em_count;
                max_pos = j;
            }
        }
        if (find_tag(sentence[b]) == max_pos)
        {
            *success += 1;
        }
        *total += 1;
    }
}

void eval_sent(int *start, int *end, int *tags, int**transition, char **sentence, int len, size_t *success, size_t *total)
{
    double **prob = (double **) malloc(sizeof(double *) * len);
    int **back = (int **) malloc(sizeof(int *) * len);
    for (int i = 0; i < len; i++)
    {
        prob[i] = (double *) malloc(sizeof(double) * (POS_NUM + 1));
        back[i] = (int *) malloc(sizeof(int) * (POS_NUM + 1));
    }
    for (int k = 0; k < POS_NUM; k++)
    {
        if (tags[k] != 0)
        {
            prob[0][k] = log2(((double) start[k]) / ((double) tags[k]));
        }
        else
        {
            prob[0][k] = log2(0);
        }
    }

    for (int i = 0; i < POS_NUM + 1; i++)
    {
        double max_last = -1.0 * DBL_MAX;
        int max_index = 0;
        for (int j = 0; j < POS_NUM + 1; j++)
        {
            double curr = prob[0][j] + log2(((double) transition[j][i]) / ((double) tags[j]));
            if (emission.find(sentence[0]) == emission.end() || emission[sentence[0]]->count <= REMOVE_THRESH)
            {
                curr +=  log2(((double)emission[unknown]->pos[j]) / ((double) tags[j]));
            }
            else
            {
                curr +=  log2(((double)emission[sentence[0]]->pos[j]) / ((double) tags[j]));
            }
            if (curr > max_last)
            {
                max_last = curr;
                max_index = j;
            }
        }
        back[0][i] = max_index;
    }

    for (int i = 1; i < len; i++)
    {
        int a = (2 * i);
        int b = (2 * i) + 1;

        for (int j = 0; j < POS_NUM + 1; j++)
        {
            double max_last = -1.0 * DBL_MAX;
            int max_index = 0;
            for (int k = 0; k < POS_NUM + 1; k++)
            {
                double curr = prob[i - 1][k];
                curr += log2(((double) transition[k][j]) / ((double) tags[k]));    
                double em_count = 0;
                if (emission.find(sentence[a]) == emission.end() || emission[sentence[a]]->count <= REMOVE_THRESH)
                {
                    em_count = ((double)emission[unknown]->pos[k]) / ((double) emission[unknown]->count);
                }
                else
                {
                    em_count = ((double)emission[sentence[a]]->pos[k]) / ((double) emission[sentence[a]]->count);
                }
                curr += log2(em_count);
                if (curr > max_last)
                {
                    max_last = curr;
                    max_index = k;
                }
            }
            prob[i][j] = max_last;
            back[i][j] = max_index;
            if (i == len - 1)
            {
                prob[i][j] += log2(((double) end[j]) / ((double) tags[j]));
            }
        }
    }
    double max_final = -1.0 * DBL_MAX;
    int max_back = 0;
    for (int i = 0; i < POS_NUM; i++)
    {
        if (prob[len - 1][i] > max_final)
        {
            max_back = i;
            max_final = prob[len - 1][i];
        }
    }
    int back_pt = max_back;
    for (int i = len - 1; i >= 0; i--)
    {
        if (find_tag(sentence[2 * i + 1]) == back_pt)
        {
            *success += 1;
        }
        *total += 1;
        if (i != 0)
        {
            back_pt = back[i - 1][back_pt];
        }
    }
}

void filter()
{
    word *w = (word *) malloc(sizeof(word));
    w->count = 0;
    for (int i = 0; i < POS_NUM + 1; i++)
    {
        w->pos[i] = 0;
    }
    emission[unknown] = w;
    printf("WORDS: %d\n", emission.size());
    for (auto w1 : emission)
    {
        if ((w1.second)->count <= REMOVE_THRESH)
        {
            emission[unknown]->count += (w1.second)->count;
            for (int i = 0; i < POS_NUM + 1; i++)
            {
                emission[unknown]->pos[i] += (w1.second)->pos[i];
            }
        }
    }
}

void train(FILE *in, int *start, int *end, int *tags, int **transition)
{
    int sen_count = 0;
    while (train_sentence(in, start, end, tags, transition) == 0)
    {
        sen_count++;
    }
    fprintf(stdout, "TOTAL SENTENCES: %d\n", sen_count);
}

char **get_sentence(FILE *in, int *len)
{
    char *line;
    int size = 0;
    int cap = 4;
    char **parsed = (char **) malloc(sizeof(char *) * cap);
    int length = -2;
    while ((line = get_line(in, &length)) != NULL)
    {
        if (line[0] == '#' || line[1] == '-' || line[2] == '-')
        {
            free(line);
            continue;
        }
        char **tokens = split(line, length);
        char *p1 = (char *) malloc(strlen(tokens[0]) + 1);
        char *p2 = (char *) malloc(strlen(tokens[2]) + 1);
        p1 = strcpy(p1, tokens[0]);
        p2 = strcpy(p2, tokens[2]);
        parsed[size++] = p1;
        parsed[size++] = p2;
        if (size == cap)
        {
            cap *= 2;
            parsed = (char **) realloc(parsed, sizeof(char *) * cap);
        }
    }
    *len = size / 2;
    return parsed;
}

int train_sentence(FILE *in, int *start, int *end, int *tags, int **transition)
{
    int length = -2;
    char *line;
    bool first = true;
    int tag_no = 0;
    int tag_last = 0;
    int word_no = 0;
    while ((line = get_line(in, &length)) != NULL)
    {
        word_no++;
        if (line[0] == '#' || line[1] == '-' || line[2] == '-')
        {
            free(line);
            continue;
        }
        char **tokens = split(line, length);
        tag_no = find_tag(tokens[2]);
        tags[tag_no] += 1;
        auto word_em = emission.find(tokens[0]);
        if (word_em == emission.end())
        {
            word *new_word = (word *) malloc(sizeof(word));
            new_word->count = 1;
            for (int i = 0; i < POS_NUM; i++)
            {
                new_word->pos[i] = 0;
                if (i == tag_no)
                {
                    new_word->pos[i] += 1;
                }
            }
            emission[tokens[0]] = new_word;
        }
        else
        {
            (word_em->second)->count += 1;
            (word_em->second)->pos[tag_no] += 1;
        }
        if (first)
        {
            if (tag_no > -1 && tag_no < POS_NUM)
            {
                start[tag_no] += 1;
            }
            first = !first;
        }
        else
        {
            transition[tag_last][tag_no] += 1;
        }
        tag_last = tag_no;
        free(line);
    }
    if (first)
    {
        if (word_no != 0)
        {
            return 0;
        }
        return -1;
    }
    end[tag_no] += 1;
    return 0;
}

char **split(char *line, int length)
{
    int line_item = -1;
    int token_size = 0;
    int token_cap = 16;
    char **tokenized = (char **) malloc(sizeof(char *) * 3);
    tokenized[0] = (char *) malloc(sizeof(char) * token_cap);
    tokenized[1] = (char *) malloc(sizeof(char) * token_cap);
    tokenized[2] = (char *) malloc(sizeof(char) * token_cap);
    // printf("%s\n", line);
    for (int i = 0; i < length; i++)
    {
        if (isspace(line[i]))
        {
            if (line_item > -1 && line_item < 3)
            {

                tokenized[line_item][token_size] = '\0';
            }
            token_size = 0;
            token_cap = 16;
            line_item++;
            if (line_item == 3)
            {
                return tokenized;
            }
        }
        else if (line_item > -1 && line_item < 3)
        {
            tokenized[line_item][token_size++] = tolower(line[i]);
            if (token_size == token_cap)
            {
                token_cap *= 2;
                tokenized[line_item] = (char *) realloc(tokenized[line_item], sizeof(char) * token_cap);
            }
        }
    }
    free(tokenized[0]);
    free(tokenized[1]);
    free(tokenized[2]);
    free(tokenized);
    return NULL;
}

char *get_line(FILE *in, int *len)
{
    int line_size = 0;
    int line_cap = 256;
    char *line = (char *) malloc(sizeof(char) * line_cap);
    char c = 'f';
    while ((fscanf(in, "%c", &c)) != EOF && c != '\n')
    {
        line[line_size++] = c;
        if (line_size == line_cap)
        {
            line_cap *= 2;
            line = (char *) realloc(line, sizeof(char) * line_cap);
        }
    }
    if (c == '\n' && line_size != 0)
    {
        line[line_size] = '\0';
        *len = line_size;
        return line;
    }
    free(line);
    *len = -1;
    return NULL;
}

int find_tag(char *tag)
{
    for (int i = 0; i < POS_NUM; i++)
    {
        int comp = strcmp(tag, POS_STANDARD[i]);
        if (comp == 0)
        {
            return i;
        }
        else if (comp < 0)
        {
            break;
        }
    }
    return POS_NUM;
}
