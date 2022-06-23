# models.py

import numpy as np
import collections

import torch
import torch.nn as nn
import random
from torch.nn.modules import dropout

from torch.nn.modules.loss import MSELoss

#####################
# MODELS FOR PART 1 #
#####################

class ConsonantVowelClassifier(object):
    def predict(self, context):
        """
        :param context:
        :return: 1 if vowel, 0 if consonant
        """
        raise Exception("Only implemented in subclasses")


class FrequencyBasedClassifier(ConsonantVowelClassifier):
    """
    Classifier based on the last letter before the space. If it has occurred with more consonants than vowels,
    classify as consonant, otherwise as vowel.
    """
    def __init__(self, consonant_counts, vowel_counts):
        self.consonant_counts = consonant_counts
        self.vowel_counts = vowel_counts

    def predict(self, context):
        # Look two back to find the letter before the space
        if self.consonant_counts[context[-1]] > self.vowel_counts[context[-1]]:
            return 0
        else:
            return 1


class RNNClassifier(nn.Module):
    def __init__(self, dict_size, input_size, hidden_size, dropout, vocab_index, rnn_type = 'lstm') -> None:
        super(RNNClassifier, self).__init__()
        self.word_embedding = nn.Embedding(dict_size, input_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.vocab_index = vocab_index
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, dropout=dropout)
        self._linear = nn.Linear(hidden_size, 2)
        self._softmax = nn.Softmax()
        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.rnn.weight_hh_l0, a=-1.0/np.sqrt(self.hidden_size), b=1.0/np.sqrt(self.hidden_size))
        nn.init.uniform_(self.rnn.weight_ih_l0, a=-1.0 / np.sqrt(self.hidden_size),
                        b=1.0 / np.sqrt(self.hidden_size))
        nn.init.uniform_(self.rnn.bias_hh_l0, a=-1.0 / np.sqrt(self.hidden_size),
                        b=1.0 / np.sqrt(self.hidden_size))
        nn.init.uniform_(self.rnn.bias_ih_l0, a=-1.0 / np.sqrt(self.hidden_size),
                        b=1.0 / np.sqrt(self.hidden_size))

    def forward(self, input):
        embedded_input = self.word_embedding(input)
        embedded_input = embedded_input.view(len(embedded_input), 1, -1)
        init_state = (torch.from_numpy(np.zeros(self.hidden_size)).unsqueeze(0).unsqueeze(1).float(),
                      torch.from_numpy(np.zeros(self.hidden_size)).unsqueeze(0).unsqueeze(1).float())
        output, (hidden_state, cell_state) = self.rnn(embedded_input, init_state)
        pred = self._linear(hidden_state[-1][-1])
        return self._softmax(pred)

    def predict(self, context):
        input_seq = []
        for i in range(len(context)):
            input_seq.append(self.vocab_index.index_of(context[i]))
        pred = self(torch.Tensor(input_seq).long())
        if pred[0] > 0.5:
            return 0
        else:
            return 1


def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)


def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_cons_exs: list of strings followed by consonants
    :param train_vowel_exs: list of strings followed by vowels
    :param dev_cons_exs: list of strings followed by consonants
    :param dev_vowel_exs: list of strings followed by vowels
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNClassifier instance trained on the given data
    """

    EPOCH = 5

    model = RNNClassifier(len(vocab_index), 5, 20, 0.5, vocab_index)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    
    input_seq = []
    input_label = []
    for letters in train_cons_exs:
        tmp = []
        for i in range(len(letters)):
            tmp.append(vocab_index.index_of(letters[i]))
        input_seq.append(tmp)
        input_label.append(0)
    for letters in train_vowel_exs:
        tmp = []
        for i in range(len(letters)):
            tmp.append(vocab_index.index_of(letters[i]))
        input_seq.append(tmp)
        input_label.append(1)

    random.Random(1).shuffle(input_seq)
    random.Random(1).shuffle(input_label)

    for epoch in range(EPOCH):
        for x, y in zip(input_seq, input_label):
            prob = model(torch.Tensor(x).long())
            loss = torch.neg(torch.log(prob[y]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model

#####################
# MODELS FOR PART 2 #
#####################


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e
        :param context: a single character to score
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class RNNLanguageModel(nn.Module):
    def __init__(self, dict_size, input_size, hidden_size, dropout, vocab_index, rnn_type = 'lstm') -> None:
        super(RNNLanguageModel, self).__init__()
        self.word_embedding = nn.Embedding(dict_size, input_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.vocab_index = vocab_index
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, dropout=dropout)
        self._linear = nn.Linear(hidden_size, 27)
        self._softmax = nn.Softmax()
        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.rnn.weight_hh_l0, a=-1.0/np.sqrt(self.hidden_size), b=1.0/np.sqrt(self.hidden_size))
        nn.init.uniform_(self.rnn.weight_ih_l0, a=-1.0 / np.sqrt(self.hidden_size),
                        b=1.0 / np.sqrt(self.hidden_size))
        nn.init.uniform_(self.rnn.bias_hh_l0, a=-1.0 / np.sqrt(self.hidden_size),
                        b=1.0 / np.sqrt(self.hidden_size))
        nn.init.uniform_(self.rnn.bias_ih_l0, a=-1.0 / np.sqrt(self.hidden_size),
                        b=1.0 / np.sqrt(self.hidden_size))

    def forward(self, input):
        embedded_input = self.word_embedding(input)
        embedded_input = embedded_input.view(len(embedded_input), 1, -1)
        init_state = (torch.from_numpy(np.zeros(self.hidden_size)).unsqueeze(0).unsqueeze(1).float(),
                      torch.from_numpy(np.zeros(self.hidden_size)).unsqueeze(0).unsqueeze(1).float())
        output, (hidden_state, cell_state) = self.rnn(embedded_input, init_state)
        prob = self._linear(output)
        s = torch.nn.Softmax(dim = 2)
        prob = s(prob)
        return torch.log(prob)

    def get_next_char_log_probs(self, context):
        input_seq = []
        for letter in context:
            input_seq.append(self.vocab_index.index_of(letter))
        log_probs = self(torch.Tensor(input_seq).long())
        log_probs = log_probs.detach().numpy()
        return log_probs[-1][0]

    def get_log_prob_sequence(self, next_chars, context):
        input_seq = []
        sum_log_prob = 0
        for letter in context:
            input_seq.append(self.vocab_index.index_of(letter))
        log_probs = self(torch.Tensor(input_seq).long())
        log_probs = log_probs.detach().numpy()
        log_probs = log_probs[-1][0]
        for nc in next_chars:
            sum_log_prob += log_probs[self.vocab_index.index_of(nc)]
            context = context + nc
            log_probs = self.get_next_char_log_probs(context)
        return sum_log_prob


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """
    EPOCH = 2
    chunk_size = 10

    data = []
    label = []

    for i in range(int(len(train_text) / chunk_size) - 1):
        tmp = []
        tmp_label = []
        for j in range(chunk_size):
            tmp.append(vocab_index.index_of(train_text[i * chunk_size + j]))
            tmp_label.append(vocab_index.index_of(train_text[i * chunk_size + j + 1]))
        data.append(tmp)
        label.append(tmp_label)

    model = RNNLanguageModel(len(vocab_index), 10, 20, 0.5, vocab_index)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.004)
    
    for epoch in range(EPOCH):
        random.Random(1).shuffle(data)
        random.Random(1).shuffle(label)
        for i in range(len(data) - 1):
            prob = model(torch.Tensor(data[i]).long())
            loss = 0
            for j in range(chunk_size):
                loss += torch.neg(prob[j][0][label[i][j]])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model
