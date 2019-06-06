import argparse
import os
from loaders import SequenceLoader, VocabularyLoader, UserNameLoader
from trainers import train_from_scratch, resume_training
from data_generator import DataSetConfig, DataSet


loaders_dict = {
    'SequenceLoader': SequenceLoader,
    'VocabularyLoader': VocabularyLoader,
    'UserNameLoader': UserNameLoader
}


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loader_class', type=str, default='UserNameLoader')

    parser.add_argument('--sequences_path', type=str, default='sequences.txt')

    parser.add_argument('--num_sequences', type=int, default=100000)
    parser.add_argument('--max_len', type=int, default=16)
    parser.add_argument('--sentinel', type=str, default=' ')

    parser.add_argument('--units', type=int, default=512)
    parser.add_argument('--layers', type=int, default=3)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)

    parser.add_argument('--save_path', type=str, default='incognito_model.h5')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()

    LoaderClass = loaders_dict[args.loader_class]

    loader = LoaderClass(args.sequences_path, max_sequences=args.num_sequences)

    sequences = loader.get(args.num_sequences)

    config = DataSetConfig(max_len=args.max_len, alphabet_size=128,
                           filler=args.sentinel)

    data_set = DataSet(sequences=sequences, data_set_config=config)

    model_path = args.save_path

    if os.path.isfile(model_path):
        resume_training(data_set=data_set, batch_size=args.batch_size,
                        epochs=args.epochs, model_path=model_path)
    else:
        train_from_scratch(data_set=data_set, units=args.units,
                           layers=args.layers, batch_size=args.batch_size,
                           epochs=args.epochs, save_path=model_path)
