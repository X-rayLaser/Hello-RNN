import os
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from models import CharacterPredictor, Architecture


def model_exists(path):
    return os.path.isfile(path)


def try_loading_model(path):
    try:
        return CharacterPredictor.from_existing_model(path)
    except Exception as e:
        print('Cannot continue training.'
              'Did you change architecture?\n')
        raise e


def train(predictor, data_set, batch_size, epochs, save_path):
    training_generator, validation_generator = data_set.create_generators(
        batch_size=batch_size
    )

    train_gen = predictor.wrap_generator(training_generator.mini_batches())
    dev_gen = predictor.wrap_generator(validation_generator.mini_batches())

    model = predictor.model
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    saving_callback = ModelCheckpoint(filepath=save_path,
                                      save_weights_only=True, period=1)

    model.fit_generator(train_gen,
                        steps_per_epoch=training_generator.steps_per_epoch(),
                        epochs=epochs,
                        validation_data=dev_gen,
                        validation_steps=validation_generator.steps_per_epoch(),
                        callbacks=[saving_callback])


def train_from_scratch(data_set, units=64, layers=3, batch_size=128,
                       epochs=100, save_path='LSTM_model.h5'):
    data_set_config = data_set.data_set_config

    arch = Architecture(time_steps=data_set_config.max_len,
                        hidden_states=units,
                        num_features=data_set_config.alphabet_size,
                        layers=layers)

    predictor = CharacterPredictor(arch, save_path=save_path)

    train(predictor=predictor, data_set=data_set, batch_size=batch_size,
          epochs=epochs, save_path=save_path)


def resume_training(data_set, batch_size=128, epochs=100, model_path='LSTM_model.h5'):
    predictor = try_loading_model(model_path)

    train(predictor=predictor, data_set=data_set, batch_size=batch_size,
          epochs=epochs, save_path=model_path)
