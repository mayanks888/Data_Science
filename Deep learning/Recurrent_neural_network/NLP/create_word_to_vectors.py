import tensorflow as tf
import  nltk
from nltk import word_tokenize,sent_tokenize
import gensim
from gensim.models.word2vec import  Word2Vec
import  pandas as pd
from sklearn.manifold import TSNE

# import bokeh.

# from natural language tokenizer we will download punkt, this is english sentenses tokeniser, convert raw text into token
nltk.download('punkt')

#llaoding our datasets gutenberg whilch contain 16 books
# nltk.download('gutenberg')
from nltk.corpus import gutenberg
print(gutenberg.fileids())#to read title of books
print(len(gutenberg.fileids())) #len of books

get_data=nltk.sent_tokenize(gutenberg.raw())#start taking sentense from all the 18 books seprated by on ".",fullstop
# convert all book line into list based on fullstop
print(get_data[1])#sentenses at location [1]
print(len(get_data))#total sentenses len

sep_word=word_tokenize(get_data[1])# seprate senstense no[1] into words
print(sep_word)
print(len(sep_word))#no of words in a sentenses

#now for better breaking of sentenses into words we can use inbuilts library from nltk
gberg_sents = gutenberg.sents()
print(gberg_sents[0:5])#take first five sentenses
print(len(gberg_sents))
print(gberg_sents[1])

model=Word2Vec(sentences=gberg_sents,size=64,sg=1, window=10, min_count=5,seed=42,iter=5)
sentences=gberg_sents# input data of sentense sep by words
size=64#no of dimension for data should create into
sg=1# sg ({0, 1}, optional) – Training algorithm: 1 for skip-gram; otherwise CBOW.
window=10 # take 10 neighbour word from left and 1= from right that makes a batch of 20
min_count=5 #only appy word to vec algorithm if word is present more than 5 min in all the complere list of sentenses
# or Ignores all words with total frequency lower than thi
seed=42 #take random data at seed 42
iter=5 #  no of iteration/epochs to be run
model.save("my_gutenberg_word_to_vec.w2v")#save your model
