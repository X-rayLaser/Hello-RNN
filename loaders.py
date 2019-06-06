from wordfreq import word_frequency


class SequenceLoader:
    def __init__(self, path, max_sequences):
        self._path = path
        self._sequences = []
        self._max_seq = max_sequences

        self.load()

    def load(self):
        sequences = []
        with open(self._path) as f:
            for index, line in enumerate(f):
                s = line.rstrip()
                seq = self.row_to_sequence(index, s)
                if len(sequences) < self._max_seq and self.keep(index, seq):
                    sequences.append(seq)

        self._sequences = self.post_process(sequences)

    def get(self, n):
        return self._sequences[:n]

    def keep(self, index, seq):
        return True

    def row_to_sequence(self, index, row):
        return row

    def post_process(self, sequences):
        return sequences


class VocabularyLoader(SequenceLoader):
    def post_process(self, sequences):
        words_n_freqs = []
        for w in sequences:
            f = word_frequency(word=w, lang='en')
            words_n_freqs.append((w, f))

        sorted_tuples = sorted(words_n_freqs, key=lambda t: t[1], reverse=True)
        return [w for w, f in sorted_tuples]


class UserNameLoader(SequenceLoader):
    def valid_text(self, text):
        for ch in text:
            if ord(ch) < 0 or ord(ch) >= 128:
                return False

        return True

    def keep(self, index, seq):
        return index > 0 and self.valid_text(seq)

    def row_to_sequence(self, index, row):
        return row.split(',')[0]
