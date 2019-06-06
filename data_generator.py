import random
import numpy as np
from keras.utils import to_categorical


def char_to_vec(char, alphabet_size):
    ascii_code = ord(char)

    if ascii_code >= alphabet_size:
        ascii_code = ord('?')
    return to_categorical(ascii_code, num_classes=alphabet_size)


def vec_to_char(v):
    ascii_code = v.argmax()
    return chr(ascii_code)


class CharacterTable:
    def __init__(self, corpus):
        self._characters = list(set(list(corpus)))

        self._indices = {}

        for index, ch in enumerate(self._characters):
            self._indices[ch] = index

    @property
    def size(self):
        return len(self._characters)

    def encode(self, character):
        index = self._indices[character]
        return to_categorical(index, num_classes=self.size)

    def decode(self, pmf, stochastic=True):
        if stochastic:
            return np.random.choice(self._characters, p=pmf)

        index = pmf.argmax()
        return self._characters[index]


def seq_to_matrix(seq, alphabet_size, padded_len):
    a = np.zeros((padded_len, alphabet_size))
    for i in range(len(seq)):
        ch = seq[i]
        a[i, :] = char_to_vec(ch, alphabet_size)

    return a


class DataSetConfig:
    def __init__(self, max_len, alphabet_size, filler):
        self.max_len = max_len
        self.alphabet_size = alphabet_size
        self.filler = filler


class DataSet:
    def __init__(self, sequences, data_set_config, split_ratio=0.9):
        self._sequences = sequences
        self._config = data_set_config

        m = len(sequences)

        m_train = int(round(m * split_ratio))
        self._train_set = sequences[:m_train]
        self._val_set = sequences[m_train:]

    def create_generators(self, batch_size=128):
        train_gen = DataGenerator(self._config, self.training_set, batch_size)
        val_gen = DataGenerator(self._config, self.validation_set, batch_size)
        return train_gen, val_gen

    @property
    def training_set(self):
        return self._train_set

    @property
    def validation_set(self):
        return self._val_set

    @property
    def data_set_config(self):
        return self._config


class DataGenerator:
    def __init__(self, data_set_config, sequences, batch_size=128):
        self._config = data_set_config
        self._batch_size = batch_size
        self._sequences = sequences

    def steps_per_epoch(self):
        set_size = len(self._sequences)
        return int(set_size / self._batch_size) + 1

    def mini_batches(self):
        words = self._sequences

        Tx = self._config.max_len

        gen = self.words_generator(words)

        for incoming, outcoming in gen:
            x_batch = self.examples_from_words(incoming)
            y = self.examples_from_words(outcoming)

            y_batch = []
            for t in range(Tx):
                y_batch.append(y[:, t, :])

            yield x_batch, y_batch

    def words_generator(self, words):
        while True:
            for i in range(0, len(words), self._batch_size):
                batch_in = []
                batch_out = []
                for word in words[i:i + self._batch_size]:
                    batch_in.append(word)
                    batch_out.append(word[1:] + ' ')
                yield batch_in, batch_out

            random.shuffle(words)

    def examples_from_words(self, words):
        Tx = self._config.max_len

        nx = self._config.alphabet_size
        x_train = np.zeros((len(words), Tx, nx), dtype=np.uint8)

        for index, word in enumerate(words):
            while len(word) < Tx:
                word += self._config.filler

            word = word[:Tx]
            x = seq_to_matrix(word, nx,
                              padded_len=Tx)
            x_train[index] = x

        return x_train
