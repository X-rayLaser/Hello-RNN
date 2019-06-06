import numpy as np
from models import CharacterPredictor
from data_generator import char_to_vec


def get_model(path):
    predictor = CharacterPredictor.from_existing_model(path)
    return predictor.get_inference_model()


class Sampler:
    def __init__(self, inference_model, prefix, alphabet_size):
        self._model = inference_model
        self._prefix = prefix
        self._alphabet_size = alphabet_size

    def sample(self, previous_character):
        x = char_to_vec(previous_character, alphabet_size=self._alphabet_size)
        return self._predict_character(x)

    def auto_complete(self, num_results=100, maxlen=100, sentinel=' '):
        outputs = []
        for _ in range(num_results):
            self._model.reset_state()

            next_char = self._feed_sequence()

            output = self._prefix

            for _ in range(maxlen):
                if next_char == sentinel:
                    break

                output += next_char

                next_char = self.sample(next_char)

            outputs.append(output)

        return outputs

    def _feed_sequence(self):
        if not self._prefix:
            x = np.zeros(self._alphabet_size)
            return self._predict_character(x)

        predicted_char = 'a'
        for ch in self._prefix:
            predicted_char = self.sample(ch)

        return predicted_char

    def _predict_character(self, x):
        indices = list(range(self._alphabet_size))

        x = x.reshape((1, 1, self._alphabet_size))

        y_hat = inference_model.predict(x)
        y_hat = y_hat[0]
        index = np.random.choice(indices, p=y_hat)
        return chr(index)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--maxlen', type=int, default=50)
    parser.add_argument('--sentinel', type=str, default=' ')
    parser.add_argument('--num_results', type=int, default=100)
    parser.add_argument('--weights_path', type=str, default='english_words_5000.h5')

    args = parser.parse_args()

    alphabet_size = 128

    inference_model = get_model(path=args.weights_path)

    while True:
        prefix = input('Enter first n characters:\n')

        sampler = Sampler(inference_model=inference_model, prefix=prefix,
                          alphabet_size=alphabet_size)

        generated_sequences = sampler.auto_complete(num_results=args.num_results,
                                                    maxlen=args.maxlen,
                                                    sentinel=args.sentinel)
        print('Generated sequences:')
        print(generated_sequences)
