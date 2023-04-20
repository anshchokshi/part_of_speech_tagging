import os
import sys
import argparse
from collections import defaultdict

import numpy as np


class HMM:
    """
    State representing the hidden markov model.
    """

    def __init__(self, training_list):
        self.observations = []
        self.word_index = {}
        self.hidden_states = []
        self.transition_prob = np.array([])
        self.observation_prob = np.array([])
        self.initial_prob = np.array([])
        self.initial_count = {}
        self.observation_count = {}
        self.sentences = []
        self.training_list = training_list
        self.tags = ["AJ0", "AJC", "AJS", "AT0", "AV0", "AVP", "AVQ", "CJC", "CJS", "CJT", "CRD",
                     "DPS", "DT0", "DTQ", "EX0", "ITJ", "NN0", "NN1", "NN2", "NP0", "ORD", "PNI",
                     "PNP", "PNQ", "PNX", "POS", "PRF", "PRP", "PUL", "PUN", "PUQ", "PUR", "TO0",
                     "UNC", 'VBB', 'VBD', 'VBG', 'VBI', 'VBN', 'VBZ', 'VDB', 'VDD', 'VDG', 'VDI',
                     'VDN', 'VDZ', 'VHB', 'VHD', 'VHG', 'VHI', 'VHN', 'VHZ', 'VM0', 'VVB', 'VVD',
                     'VVG', 'VVI', 'VVN', 'VVZ', 'XX0', 'ZZ0', 'AJ0-AV0', 'AJ0-VVN', 'AJ0-VVD',
                     'AJ0-NN1', 'AJ0-VVG', 'AVP-PRP', 'AVQ-CJS', 'CJS-PRP', 'CJT-DT0', 'CRD-PNI', 'NN1-NP0', 'NN1-VVB',
                     'NN1-VVG', 'NN2-VVZ', 'VVD-VVN', 'AV0-AJ0', 'VVN-AJ0', 'VVD-AJ0', 'NN1-AJ0', 'VVG-AJ0', 'PRP-AVP',
                     'CJS-AVQ', 'PRP-CJS', 'DT0-CJT', 'PNI-CRD', 'NP0-NN1', 'VVB-NN1', 'VVG-NN1', 'VVZ-NN2', 'VVN-VVD']

    def setup_training(self):
        """
        function to parse through the training file and take not of all the observations and hidden states.
        """

        for training_file in self.training_list:
            with open(training_file, 'r') as f:
                training_data = f.readlines()
            for line in training_data:
                line = line.split(':')
                if len(line) == 2:
                    word = line[0].strip()
                    tag = line[1].strip()

                    self.observations.append(word)
                    self.hidden_states.append(tag)
                    if tag in self.initial_count:
                        self.initial_count[tag] += 1
                    else:
                        self.initial_count[tag] = 1
                    if word in self.observation_count:
                        self.observation_count[word] += 1
                    else:
                        self.observation_count[word] = 1

                else:
                    word = line[0].strip()
                    if word == '':
                        word = ':'
                    tag = line[2].strip()

                    self.observations.append(word)
                    self.hidden_states.append(tag)
                    if tag in self.initial_count:
                        self.initial_count[tag] += 1

                    else:
                        self.initial_count[tag] = 1

                    if word in self.observation_count:
                        self.observation_count[word] += 1
                    else:
                        self.observation_count[word] = 1

    def form_sentences(self):
        """
        from all the observed words, we form sentences in this function by separating them based on various
        punctuations.
        """
        i = 0
        sentence = []
        all_sentences = []
        for word in self.observations:
            if word not in ['.', '?', '!', '...']:
                temp = (word, self.hidden_states[i])
                sentence.append(temp)
                i += 1
            else:
                sentence.append((word, self.hidden_states[i]))
                all_sentences.append(sentence)
                sentence = []
                i += 1

        self.sentences = all_sentences

    def count_initial_probability(self):
        """
        function to count initial probability and set up a 1xN matrix.
        N = number of tags
        """
        first_tag_count = {}

        for sentence in self.sentences:
            first_word, pos_tag = sentence[0]
            if pos_tag in first_tag_count:
                first_tag_count[pos_tag] += 1
            else:
                first_tag_count[pos_tag] = 1

        initial_counts = np.zeros((1, len(self.tags)))
        for key, value in first_tag_count.items():
            prob = value / len(self.sentences)
            initial_counts[0, self.tags.index(key)] = prob

        self.initial_prob = initial_counts

    def count_pairs(self):
        """
        from all the hidden states, track all the hidden states connected to each other.
        This help us find probability of tag1 coming next given that tag2 is observed.
        """

        pair_counts = np.ones((len(self.tags), len(self.tags)))
        p_counts = np.zeros((len(self.tags)))

        for i in range(1, len(self.hidden_states)):
            prior, current = self.hidden_states[i - 1], self.hidden_states[i]
            word = self.observations[i - 1]
            if word not in ['.', '?', '!', '...']:
                row_idx = self.tags.index(prior)
                col_idx = self.tags.index(current)
                pair_counts[row_idx][col_idx] += 1
                p_counts[row_idx] += 1
        return pair_counts, p_counts

    def count_transition_probability(self):
        """
        function to count transitional probability and set up a NxN matrix.
        N = number of tags
        """
        transitional_prob, prior_counts = self.count_pairs()
        for row, count in enumerate(prior_counts):
            if count != 0:
                final = (count + len(self.tags))
                transitional_prob[row] /= final

        self.transition_prob = transitional_prob

    def count_word_tag_pairs(self):
        """
        function to count all word and tag pairs. This helps us to calculate the observational probability.
        only unique words are added to the dictionary and the index is noted down.
        """
        unique_words = {}
        index = 0
        for i in range(len(self.observations)):
            word = self.observations[i]
            if word not in unique_words:
                unique_words[word] = index
                index += 1
        self.word_index = unique_words

        observation_counts = np.zeros((len(self.tags), len(unique_words)))

        for i in range(len(self.observations)):
            word, tag = self.observations[i], self.hidden_states[i]
            word_index = unique_words[word]
            tag_index = self.tags.index(tag)
            observation_counts[tag_index][word_index] += 1

        return observation_counts

    def count_observed_probability(self):
        """
        function to count observed probability and set up a NxM matrix.
        N = number of tags. M = number of unique words
        """
        word_counts = self.count_word_tag_pairs()
        for tag, count in self.initial_count.items():
            tag_index = self.tags.index(tag)
            word_counts[tag_index] /= count

        self.observation_prob = word_counts


def get_sentences(test_file):
    """
    function to get sentences for the test file.
    """

    all_sentences = []

    with open(test_file, 'r') as f:
        training_data = f.readlines()
    sentence = []
    for line in training_data:
        word = line.strip()
        if word not in ['.', '?', '!', '...']:
            sentence.append(word)
        else:
            sentence.append(word)
            all_sentences.append(sentence)
            sentence = []
    return all_sentences


def viterbi(hmm, test_sentence):
    """
    viterbi algorithm to find the best tags sequence for the test sentence. We use all the probabilities
    we calculated before in the HMM state here.
    """

    prob = np.zeros((len(test_sentence), len(hmm.tags)))
    prev = np.zeros((len(test_sentence), len(hmm.tags)))

    for i in range(len(hmm.tags)):
        if test_sentence[0] in hmm.word_index:
            e0 = hmm.word_index[test_sentence[0]]
            prob[0][i] = hmm.initial_prob[0, i] * hmm.observation_prob[i, e0]
            prev[0][i] = None
        else:
            prob[0][i] = hmm.initial_prob[0, i]
            prev[0][i] = None

    for t in range(1, len(test_sentence)):
        for i in range(len(hmm.tags)):

            if test_sentence[t] in hmm.word_index:
                e = hmm.word_index[test_sentence[t]]
                x = np.argmax(prob[t - 1, :] * hmm.transition_prob[:, i] * hmm.observation_prob[i, e])
                prob[t][i] = prob[t - 1][x] * hmm.transition_prob[x][i] * hmm.observation_prob[i, e]
                prev[t][i] = x
            else:
                x = np.argmax(prob[t - 1, :] * hmm.transition_prob[:, i])
                prob[t][i] = prob[t - 1][x] * hmm.transition_prob[x][i]
                prev[t][i] = x

    final = np.argmax(prob[-1, :])
    sequence = [final]
    for i in range(len(test_sentence) - 1, 0, -1):
        if prev[i][sequence[-1]] is not None:
            sequence.append(int(prev[i][sequence[-1]]))
        else:
            break
    sequence.reverse()
    return sequence


def algo_setup(hmm, test_sentence):
    """
    function to recursively call viterbi algorithm for all sentences one by one.
    """

    answer = []
    for sentence in test_sentence:
        sequence = viterbi(hmm, sentence)
        answer.append(sequence)
    return answer


def print_output(tags, sequence, output_file, test_sentences):
    """
    function to printout the output once we have calculated the final sequence for all the test sentences.
    """

    sys.stdout = open(output_file, 'w')
    for i in range(len(test_sentences)):
        for j in range(len(test_sentences[i])):
            sys.stdout.write(
                test_sentences[i][j] + ' : ' + tags[sequence[i][j]])
            sys.stdout.write('\n')

    # Close file
    sys.stdout.close()
    sys.stdout = sys.__stdout__


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainingfiles",
        action="append",
        nargs="+",
        required=True,
        help="The training files."
    )
    parser.add_argument(
        "--testfile",
        type=str,
        required=True,
        help="One test file."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file."
    )
    args = parser.parse_args()

    training_list = args.trainingfiles[0]

    print("training files are {}".format(training_list))

    print("test file is {}".format(args.testfile))

    print("output file is {}".format(args.outputfile))

    print("Starting the tagging process.")
    hmm = HMM(training_list)
    hmm.setup_training()
    hmm.form_sentences()
    hmm.count_initial_probability()
    hmm.count_transition_probability()
    hmm.count_observed_probability()
    sentences = get_sentences(args.testfile)
    final_sequence = check = algo_setup(hmm, sentences)
    print_output(hmm.tags, final_sequence, args.outputfile, test_sentences)
