
# from fastai.learner import *
#
# from fastai.rnn_reg import *
# from fastai.rnn_train import *
# from fastai.nlp import *
# from fastai.lm_rnn import *
#
# import dill as pickle

import torchtext
import torch
from torchtext import vocab, data
from torchtext import datasets
from torchtext.datasets import language_modeling

# Text=data.get_tokenizer()

SEED = 1234

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

TEXT = data.Field(lower=True,tokenize='spacy')
LABEL = data.LabelField(tensor_type=torch.FloatTensor)

train, test = datasets.IMDB.splits(TEXT, LABEL,root=".data")
print('maynk')


PATH = 'data/aclImdb/'

TRN_PATH = 'train/all/'
VAL_PATH = 'test/all/'
TRN = f'{PATH}{TRN_PATH}'
VAL = f'{PATH}{VAL_PATH}'

%ls {PATH}

imdbEr.txt  imdb.vocab  models/  README  test/  tmp/  train/