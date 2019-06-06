import os
import json
import numpy as np
from keras.layers import Input, LSTM, Reshape, Dense, Lambda, Concatenate
from keras import Model


class Architecture:
    def __init__(self, time_steps, hidden_states, num_features, layers):
        self.time_steps = time_steps
        self.hidden_states = hidden_states
        self.num_features = num_features
        self.layers = layers

    @staticmethod
    def get_architecture_path(weights_path):
        base_path, _ = os.path.splitext(weights_path)
        return base_path + '.json'

    @staticmethod
    def load(path):
        arch_path = Architecture.get_architecture_path(weights_path=path)

        with open(arch_path, mode='r') as f:
            s = f.read()

        params = json.loads(s)

        return Architecture(**params)

    def save(self, path):
        arch_path = self.get_architecture_path(weights_path=path)

        arch = {
            'time_steps': self.time_steps,
            'hidden_states': self.hidden_states,
            'num_features': self.num_features,
            'layers': self.layers
        }

        s = json.dumps(arch)

        with open(arch_path, mode='w') as f:
            f.write(s)


class CharacterPredictor:
    def __init__(self, model_architecture, save_path):
        self._arch = model_architecture
        self._save_path = save_path

        self._reshapor = Reshape((1, self._arch.num_features))
        self._densor = Dense(units=self._arch.num_features, activation='softmax')

        self._cells = []
        for _ in range(self._arch.layers):
            cell = LSTM(self._arch.hidden_states, return_state=True)
            self._cells.append(cell)

        self._model = self._build()
        self._save_architecture()

    @staticmethod
    def from_existing_model(path):
        architecture = Architecture.load(path)

        predictor = CharacterPredictor(architecture, path)

        predictor.load(path)
        return predictor

    def _save_architecture(self):
        self._arch.save(self._save_path)

    def _build(self):
        X = Input(shape=(self._arch.time_steps, self._arch.num_features))

        a0 = Input(shape=(self._arch.hidden_states,), name='a0')
        c0 = Input(shape=(self._arch.hidden_states,), name='c0')

        outputs = []

        prev = [(a0, c0)] * self._arch.layers

        for t in range(self._arch.time_steps):
            x = Lambda(lambda x: X[:, t, :])(X)
            x = self._reshapor(x)

            for i in range(self._arch.layers):
                a, c = prev[i]
                cell = self._cells[i]
                a, _, c = cell(x, initial_state=[a, c])
                prev[i] = (a, c)

            last_activation = prev[-1][0]
            out = self._densor(last_activation)
            outputs.append(out)

        model = Model(inputs=[X, a0, c0], outputs=outputs)
        return model

    def _reshape_target(self, target):
        y = []
        for t in range(self._arch.time_steps):
            y.append(target[:, t, :])

        return y

    def wrap_generator(self, generator):
        cache = None
        for x_batch, y_batch in generator:
            m = len(x_batch)
            if not cache:
                a0 = np.zeros((m, self._arch.hidden_states))
                c0 = np.zeros((m, self._arch.hidden_states))
                cache = (a0, c0)
            else:
                a0, c0 = cache
                a0 = a0[:m, :]
                c0 = c0[:m, :]
            yield [x_batch, a0, c0], y_batch

    @property
    def model(self):
        return self._model

    def save(self, path='LSTM_model.h5'):
        self.model.save_weights(path)

    def load(self, path='LSTM_model.h5'):
        self._model = self._build()
        self._model.load_weights(path)

    def get_inference_model(self):
        inp = Input(shape=(None, self._arch.num_features))

        a0 = Input(shape=(self._arch.layers, self._arch.hidden_states), name='a0')
        c0 = Input(shape=(self._arch.layers, self._arch.hidden_states), name='c0')

        x = inp
        x = self._reshapor(x)

        prev_a = []
        prev_c = []

        for i in range(self._arch.layers):
            prev_a.append(
                Lambda(lambda v: a0[:, i, :])(a0)
            )
            prev_c.append(
                Lambda(lambda v: c0[:, i, :])(c0)
            )

        activations = []
        states = []

        for i in range(self._arch.layers):
            cell = self._cells[i]
            a, _, c = cell(x, initial_state=[prev_a[i], prev_c[i]])
            activations.append(a)
            states.append(c)

        y_hat = self._densor(a)

        if len(activations) == 1:
            activations_tensor = activations[0]
            states_tensor = states[0]
        else:
            activations_tensor = Concatenate()(activations)
            states_tensor = Concatenate()(states)

        a = Reshape((self._arch.layers, self._arch.hidden_states))(activations_tensor)

        c = Reshape((self._arch.layers, self._arch.hidden_states))(states_tensor)

        inference_model = Model(inputs=[inp, a0, c0], outputs=[y_hat, a, c])

        return InferenceModel(inference_model, self._arch)


class InferenceModel:
    def __init__(self, model, model_architecture):
        self._model = model
        self._architecture = model_architecture
        self._a0 = None
        self._c0 = None
        self.reset_state()

    def predict(self, x):
        y_hat, a, c = self._model.predict([x, self._a0, self._c0])
        self._a0 = a
        self._c0 = c
        return y_hat

    def reset_state(self):
        self._a0 = np.zeros((1, self._architecture.layers,
                             self._architecture.hidden_states))
        self._c0 = np.zeros((1, self._architecture.layers,
                             self._architecture.hidden_states))
