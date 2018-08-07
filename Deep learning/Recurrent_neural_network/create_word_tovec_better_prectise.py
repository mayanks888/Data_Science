# the initial block is copied from creating_word_vectors_with_word2vec.ipynb
import nltk
from nltk import word_tokenize, sent_tokenize
import gensim
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE
import pandas as pd
from bokeh.io import output_notebook, output_file
from bokeh.plotting import show, figure
import string
from nltk.corpus import stopwords
from nltk.stem.porter import *
from gensim.models.phrases import Phraser, Phrases
from keras.preprocessing.text import one_hot

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('gutenberg')


from nltk.corpus import gutenberg

my_sent=gutenberg.sents()#convert into sentenses
print(my_sent[5])


# 1 practise: make all words lower case
'''my_sent1=[w.lower() for w in my_sent[1]]
print(my_sent1)

# 2 practise remove puntuation from sentesnses
stop_punt=stopwords.words('english')+list(string.punctuation)
print(stop_punt)
print(len(stop_punt))
#removel all english common word from corpus (not recommended)
print([w.lower() for w in my_sent[4] if w not in stop_punt])
print(my_sent)


#3 practise remover all plural and sigular same (son, sons, chair,chairs(not recommended)
stemmer = PorterStemmer()
[stemmer.stem(w.lower()) for w in my_sent[4] if w not in stop_punt]'''

#4 practise biagram find pair like new and york into new_york, or new delhi as new_delhi find number of biagram
# phrases = Phrases(my_sent)#convert into sentenses
# bigram = Phraser(phrases)
lower_bigram = Phraser(Phrases(my_sent, min_count=32, threshold=64))#or apply biagram with threshold
# print(bigram.phrasegrams)
print(lower_bigram.phrasegrams)
#output=(b'hath', b'shewed'): (17, 18.37949515563393), (b'mercy', b'endureth'): (41, 1161.0511603650586),
# here 17=no of instance and 18 is their score of vector similar

