# This is not RNN frame works rather it is simple dense neural network used for sentiment analysis
# Parallerl_convnets
import keras
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential,Model
from keras.layers import Input, concatenate
from keras.layers import Dense, Flatten,Dropout
from keras.layers import Embedding
from keras.layers import Conv1D,SpatialDropout1D,GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint
import os
from sklearn.metrics import roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt

# Set hyperameter

output_dir='multi_parallel_Conv/'
epochs=4
batch_size=128
n_dim=64#hidden layer neuron or dimension

n_unique_words=5000#take top  5000 most frequent  word form the corpus of imdb
n_words_to_skip=50#from markolov paper , skip top 50  frewquent word(i have confusion in this , please read)
max_review_lenth=400#to make sure all review have same no of word for ease our memory allocation
pad_type=trunc_type='pre'#pad if review has less than 100 words"pre is used to take words from end of the review
drop_emd=0.2
n_dense=256
dropout=0.2#drop out half of parameter

# Convolution layer hyperparameter parameter
filer_size=3 #here filter size is one dimensional
n_conv1=n_conv2=n_conv3=256#no of activation maps
#loading a data
keras.preprocessing.text.Tokenizer#similar to genism

(x_train,y_train),(x_valid, y_valid)=imdb.load_data(num_words=n_unique_words,skip_top=n_words_to_skip)
print(x_train[0:6])

#check length of train data and validaton data
# print('train_data is  {td} and validate data :{ted}'.format(td=len(x_train),ted=len(x_valid)))
# #check length of sentenses
# for x in x_train[0:10]:
#     print(len(x))

# convert text in numerix to string
#create word index
word_index=keras.datasets.imdb.get_word_index()
word_index={k:(v+3) for k,v in word_index.items()}
word_index['PAD']=0
word_index['START']=1
word_index['UNK']=2
# print(word_index)

index_word = {v:k for k,v in word_index.items()}
no_to_text=' '.join(index_word[id] for id in x_train[0])      #this will basically convert pick the no for sent and find the index of the word from word index
print(no_to_text)
# print(len(x_train[0]))

#Data Preprocessing

#now let us prepreosess our data as per the our requirenment explained above
x_train=pad_sequences(x_train,maxlen=max_review_lenth,padding=pad_type, truncating=trunc_type)
x_valid=pad_sequences(x_valid,maxlen=max_review_lenth,padding=pad_type, truncating=trunc_type)
# check your data preprocess
for x in x_train[0:10]:
    print(len(x))

no_to_text=' '.join(index_word[id] for id in x_train[0])      #this will basically convert pick the no for sent and find the index of the word from word index
print(no_to_text)


 #Design NN architecture
 # After this you can have keras, tensorflow, or pytorch framework for yor neural network

#
#  # KERAS sequential
# model=Sequential()
# model.add(Embedding(n_unique_words,n_dim,input_length=max_review_lenth))#it cnvert word into vector space
# # The first argument (n_unique_words) n embedded layer is the number of distinct words in the training set.
# # here n_unique word is required so as to find the lenth of one hot encoding creating for each word so as use in neural network
# # The second argument (n_dim) indicates the size of the embedding vectors
# # The input_length argumet, of course, determines the size of each input sequence.
# # model.output_shape == (None, max_review_lenth, n_dim), where None is the batch dimension
# model.add(SpatialDropout1D(drop_emd))
# model.add(Conv1D(n_conv,filer_size, activation='relu'))
# # model.add(Conv1D(filer_size=n_conv,kernel_size=filer_size, activation='relu'))
# model.add(GlobalMaxPooling1D())
# # you don't need flatten layer sicne it con1d and already in one dimensional'
# model.add(Dense(n_dense, activation='relu'))
# model.add(Dropout(dropout))
# model.add(Dense(1,activation='sigmoid'))


input_layer=Input(shape=(max_review_lenth,),dtype='int16',name="input")
emd_layer=Embedding(n_unique_words,n_dim,input_length=max_review_lenth,name='Embedded_layer')(input_layer)
drop_emb_layer=SpatialDropout1D(drop_emd,name="dropemb")(emd_layer)

conv_layer_1=(Conv1D(n_conv1,filer_size, activation='relu',name="con1_Layer"))(drop_emb_layer)
maxpool_layer1=GlobalMaxPooling1D()(conv_layer_1)

conv_layer_2=(Conv1D(n_conv1,filer_size, activation='relu',name="con2_Layer"))(drop_emb_layer)
maxpool_layer2=GlobalMaxPooling1D()(conv_layer_2)

conv_layer_3=(Conv1D(n_conv1,filer_size, activation='relu',name="con3_Layer"))(drop_emb_layer)
maxpool_layer3=GlobalMaxPooling1D()(conv_layer_3)

concat_layer=concatenate([maxpool_layer1,maxpool_layer2,maxpool_layer3])#,name="concat_layer")

Dens_layer=Dense(n_dense, activation='relu',name='dens1Layer')(concat_layer)
den_drop1_layer=Dropout(dropout)(Dens_layer)
prediction=Dense(1,activation='sigmoid',name='output')(den_drop1_layer)

model=Model(input_layer,prediction)




print(model.summary())


#compiling model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

modelcheckpoint = ModelCheckpoint(filepath=output_dir+"/weights.{epoch:02d}.hdf5")


if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# #### Train!

# In[29]:


# 84.7% validation accuracy in epoch 2
# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_valid, y_valid), callbacks=[modelcheckpoint])

#evaluating model in details

model.load_weights(output_dir+"/weights.01.hdf5") # zero-indexed

y_hat = model.predict(x_valid)

plt.hist(y_hat)
_ = plt.axvline(x=0.5, color='orange')
plt.show()
# Check roc curve
pct_auc = roc_auc_score(y_valid, y_hat)*100.0
print("{:0.2f}".format(pct_auc))


float_y_hat = []
for y in y_hat:
    float_y_hat.append(y[0])

ydf = pd.DataFrame(list(zip(float_y_hat, y_valid)), columns=['y_hat', 'y'])

print(ydf.head(10))

ydf[(ydf.y == 1) & (ydf.y_hat < 0.1)].head(10)

print(' '.join(index_word[id] for id in x_valid[927]))