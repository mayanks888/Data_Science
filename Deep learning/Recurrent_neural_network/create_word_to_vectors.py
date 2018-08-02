import tensorflow as tf
import  nltk
import gensim
from gensim.models.word2vec import  Word2Vec
import  pandas as pd
# import bokeh.


nltk.download('punkt')
#data load

nltk.download('gutenberg')
from nltk.corpus import gutenberg
print(gutenberg.fileids())
print(len(gutenberg.fileids()))

gbert_dent_token=nltk.sent_tokenize(gutenberg.raw())

gberg_sents = gutenberg.sents()
print(gbert_dent_token[0:5])

print(nltk.word_tokenize(gbert_dent_token[1]))
print(len(gutenberg.words()))

model=Word2Vec(sentences=gberg_sents,size=64,sg=1, window=10, min_count=5,seed=42)
model.save("raw_gutenberg_mdoel.w2v")

print(model['dog'])
# model =gensim.models.Word2Vec.load('raw_gutenberg_mdoel.w2v"')