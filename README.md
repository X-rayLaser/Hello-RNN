# Installation

Clone the repository in local directory,

go inside this directory,

install all dependencies via pip,

```
pip install -r requirements.txt
```

# Usage

## Sampling from pre trained models

In order to use preexisting models to sample from, execute Python 
script named 'sample.py'. You will need to pass **--weights_path** argument 
that specifies path to weights of the model. By default, the model of
english words will be used. 

After running this command,
the model will load its weights and you will see the prompt inviting
you to enter any initializing (seeding) character sequence. The model
will then (hallucinate) generate the rest of the sequence in accordance
with learned probability distribution of the next character given preceding ones.

Sample characters from english words model (generates words):
```
python sample.py --weights_path 'pre_trained_models/english_words_5000.h5'
```

Sample characters from user names model (generates user names):
```
python sample.py --weights_path 'pre_trained_models/user_names_10000.h5'
```

## Training a new model

### Preparatory steps

Prepare a text file containing sequences, one sequence per line.

Execute Python script named 'train.py'.

You will need to pass **--sequences_path** argument to specify the path to the sequences
file you prepared earlier.

To specify path where the model will be saved, pass **--save_path**
argument:

```
python train.py --sequences_path 'usernames/users.csv' --save_path 'pre_trained_models/user_names_100000.h5'
```