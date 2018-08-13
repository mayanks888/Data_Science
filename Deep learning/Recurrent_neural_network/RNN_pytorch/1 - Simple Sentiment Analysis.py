
# coding: utf-8

# # 1 - Simple Sentiment Analysis
# 
# In this series we'll be building a *machine learning* model to detect sentiment (i.e. detect if a sentence is positive or negative) using PyTorch and TorchText. This will be done on movie reviews using the IMDb dataset.
# 
# In this first notebook, we'll start very simple to understand the general concepts whilst not really caring about good results. Further notebooks will build on this knowledge, to actually get good results.
# 
# We'll be using a **recurrent neural network** (RNN) which reads a sequence of words, and for each word (sometimes called a _step_) will output a _hidden state_. We then use the hidden state for subsequent word in the sentence, until the final word has been fed into the RNN. This final hidden state will then be used to predict the sentiment of the sentence.
# 
# ![](https://i.imgur.com/VedY9iG.png)

# ## Preparing Data
# 
# One of the main concepts of TorchText is the `Field`. These define how your data should be processed. In our sentiment classification task we have to sources of data, the raw string of the review and the sentiment, either "pos" or "neg".
# 
# We use the `TEXT` field to handle the review and the `LABEL` field to handle the sentiment. 
# 
# The parameters of a `Field` specify how the data should be processed. 
# 
# Our `TEXT` field has `tokenize='spacy'`, which defines that the "tokenization" (the act of splitting the string into discrete "tokens") should be done using the [spaCy](https://spacy.io) tokenizer. If no `tokenize` argument is passed, the default is simply splitting the string on spaces.
# 
# `LABEL` is defined by a `LabelField`, a special subset of the `Field` class specifically for handling labels. We will explain the `tensor_type` argument later.
# 
# For more on `Fields`, go [here](https://github.com/pytorch/text/blob/master/torchtext/data/field.py).
# 
# We also set the random seeds for reproducibility. 

# In[1]:


import torch
from torchtext import data

SEED = 1234

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(tensor_type=torch.FloatTensor)


# Another handy feature of TorchText is that it has support for common datasets used in NLP. 
# 
# The following code automatically downloads the IMDb dataset and splits it into the canonical train/test splits as `torchtext.datasets` objects. It uses the `Fields` we have previously defined. 

# In[2]:


from torchtext import datasets

train, test = datasets.IMDB.splits(TEXT, LABEL)


# We can see how many examples are in each split by checking their length.

# In[3]:


print('len(train):', len(train))
print('len(test):', len(test))


# We can check the fields of the data, hoping that it they match the `Fields` given earlier.

# In[4]:


print('train.fields:', train.fields)


# We can also check an example.

# In[5]:


print('vars(train[0]):', vars(train[0]))


# The IMDb dataset only has train/test splits, so we need to create a validation set. We can do this with the `.split()` method. 
# 
# By default this splits 70/30, however by passing a `split_ratio` argument, we can change the ratio of the split, i.e. a `split_ratio` of 0.8 would mean 80% of the examples make up the training set and 20% make up the validation set. 
# 
# We also pass our random seed to the `random_state` argument, ensuring that we get the same train/validation split each time.

# In[6]:


import random

train, valid = train.split(random_state=random.seed(SEED))


# Again, we'll view how many examples are in each split.

# In[8]:


print('len(train):', len(train))
print('len(valid):', len(valid))
print('len(test):', len(test))


# Next, we have to build a _vocabulary_. This is a effectively a look up table where every unique word in your _dictionary_ (every word that occurs in all of your examples) has a corresponding _index_ (an integer).
# 
# ![](https://i.imgur.com/0o5Gdar.png)
# 
# We do this as our machine learning model cannot operate on strings, only numbers. Each _index_ is used to construct a _one-hot_ vector for each word. A one-hot vector is a vector where all of the elements are 0, except one, which is 1, and dimensionality is the total number of unique words in your vocabulary.
# 
# The number of unique words in our training set is over 100,000, which means that our one-hot vectors will be 100,000 dimensions! This will make training slow and possibly won't fit onto your GPU (if you're using one). 
# 
# There are two ways effectively cut down our vocabulary, we can either only take the top $n$ most common words or ignore words that appear less than $n$ times. We'll do the former, only keeping the top 25,000 words.
# 
# What do we do with words that appear in examples but we have cut from the vocabulary? We replace them with a special _unknown_ or _unk_ token. For example, if the sentence was "This film is great and I love it" but the word "love" was not in the vocabulary, it would become "This film is great and I unk it".

# In[9]:


TEXT.build_vocab(train, max_size=25000)
LABEL.build_vocab(train)


# Why do we only build the vocabulary on the training set? When testing any machine learning system you do not want to look at the test set in any way. We do not include the validation set as we want it to reflect the test set as much as possible.

# In[10]:


print('len(TEXT.vocab):', len(TEXT.vocab))
print('len(LABEL.vocab):', len(LABEL.vocab))


# Why is the vocab size 25002 and not 25000? One of the addition tokens is the _unk_ token and the other is a _pad_ token.
# 
# ![](https://i.imgur.com/TZRJAX4.png)
# 
# When we feed sentences into our model, we feed a _batch_ of them at a time, i.e. more than one at a time, and all sentences in the batch need to be the same size. Thus, to ensure each sentence in the batch is the same size, any shorter than the largest within the batch are padded.
# 
# We can also view the most common words in the vocabulary. 

# In[11]:


print(TEXT.vocab.freqs.most_common(20))


# We can also see the vocabulary directly using either the `stoi` (**s**tring **to** **i**nt) or `itos` (**i**nt **to**  **s**tring) method.

# In[12]:


print(TEXT.vocab.itos[:10])


# We can also check the labels, ensuring 0 is for negative and 1 is for positive.

# In[13]:


print(LABEL.vocab.stoi)


# The final step of preparing the data is creating the iterators.
# 
# `BucketIterator` first sorts of the examples using the `sort_key`, here we use the length of the sentences, and then partitions them into _buckets_. When the iterator is called it returns a batch of examples from the same bucket. This will return a batch of examples where each example is a similar length, minimizing the amount of padding.

# In[14]:


BATCH_SIZE = 64

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train, valid, test), 
    batch_size=BATCH_SIZE, 
    sort_key=lambda x: len(x.text), 
    repeat=False)


# ## Build the Model
# 
# The next stage is building the model that we'll eventually train and evaluate. 
# 
# There is a small amount of boilerplate code when creating models in PyTorch, note how our `RNN` class is a sub-class of `nn.Module` and the use of `super`.
# 
# Within the `__init__` we define the _layers_ of the module. Our three layers are an _embedding_ layer, our RNN, and a _linear_ layer. 
# 
# The embedding layer is used to transform our sparse one-hot vector (sparse as most of the elements are 0) into a dense embedding vector (dense as the dimensionality is a lot smaller). This embedding layer is simply a single fully connected layer. The theory is that words that have similar impact on the sentiment are mapped close together in this dense vector space. For more information about word embeddings, see [here](https://monkeylearn.com/blog/word-embeddings-transform-text-numbers/).
# 
# The RNN layer is our RNN which takes in our dense vector and the previous hidden state $h_{t-1}$, which it uses to calculate the next hidden state, $h_t$.
# 
# Finally, the linear layer takes the final hidden state and feeds it through a fully connected layer, transforming it to the correct output dimension.
# 
# ![](https://i.imgur.com/GIov3zF.png)
# 
# The `forward` method is called when we feed examples into our model.
# 
# Each batch, `x`, is a tensor of size _**[sentence length, batch size]**_. That is a batch of sentences, each having each word converted into a one-hot vector. 
# 
# You may notice that this tensor should have another dimension due to the one-hot vectors, however PyTorch conveniently stores a one-hot vector as it's index value.
# 
# The input batch is then passed through the embedding layer to get `embedded`, where now each one-hot vector is converted to a dense vector. `embedded` is a tensor of size _**[sentence length, batch size, embedding dim]**_.
# 
# `embedded` is then fed into the RNN. In some frameworks you must feed the initial hidden state, $h_0$, into the RNN, however in PyTorch, if no initial hidden state is passed as an argument it defaults to a tensor of all zeros.
# 
# The RNN returns 2 tensors, `output` of size _**[sentence length, batch size, hidden dim]**_ and `hidden` of size _**[1, batch size, embedding dim]**_. `output` is the concatenation of the hidden state from every time step, whereas `hidden` is simply the final hidden state. We verify this using the `assert` statement. Note the `squeeze` method, which is used to remove a dimension of size 1. 
# 
# Finally, we feed the last hidden state, `hidden`, through the linear layer to produce a prediction.

# In[15]:


import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):

        #x = [sent len, batch size]
        
        embedded = self.embedding(x)
        
        #embedded = [sent len, batch size, emb dim]
        
        output, hidden = self.rnn(embedded)
        
        #output = [sent len, batch size, hid dim]
        #hidden = [1, batch size, hid dim]
        
        assert torch.equal(output[-1,:,:], hidden.squeeze(0))
        
        return self.fc(hidden.squeeze(0))


# We now create an instance of our RNN class. 
# 
# The input dimension is the dimension of the one-hot vectors, which is equal to the vocabulary size. 
# 
# The embedding dimension is the size of the dense word vectors, this is usually around the square root of the vocab size. 
# 
# The hidden dimension is the size of the hidden states, this is usually around 100-500 dimensions, but depends on the vocab size, embedding dimension and the complexity of the task.
# 
# The output dimension is usually the number of classes, however in the case of only 2 classes the output value is between 0 and 1 and thus can be 1-dimensional, i.e. a single scalar.

# In[16]:


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1

model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)


# ## Train the Model

# Now we'll set up the training and then train the model.
# 
# First, we'll create an optimizer. This is the algorithm we use to update the parameters of the module. Here, we'll use _stochastic gradient descent_ (SGD). The first argument is the parameters will be updated by the optimizer, the second is the learning rate, i.e. how much we'll change the parameters by when we do an update.

# In[17]:


import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=1e-3)


# Next, we'll define our loss function. In PyTorch this is commonly called a criterion. 
# 
# The loss function here is _binary cross entropy with logits_. 
# 
# The prediction for each sentence is an unbound real number, as our labels are either 0 or 1, we want to restrict the number between 0 and 1, we do this using the _sigmoid function_, see [here](https://en.wikipedia.org/wiki/Sigmoid_function). 
# 
# We then calculate this bound scalar using binary cross entropy, see [here](https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/). 

# In[18]:


criterion = nn.BCEWithLogitsLoss()


# PyTorch has excellent support for NVIDIA GPUs via CUDA. `torch.cuda.is_available()` returns `True` if PyTorch detects a GPU.
# 
# Using `.to`, we can place the model and the criterion on the GPU. 

# In[19]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
criterion = criterion.to(device)


# Our criterion function calculates the loss, however we have to write our function to calculate the accuracy. 
# 
# This function first feeds the predictions through a sigmoid layer, squashing the values between 0 and 1, we then round them to the nearest integer. This rounds any value greater than 0.5 to 1 (a positive sentiment). 
# 
# We then calculate how many rounded predictions equal the actual labels and average it across the batch.

# In[20]:


import torch.nn.functional as F

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(F.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum()/len(correct)
    return acc


# The `train` function iterates over all examples, a batch at a time. 
# 
# `model.train()` is used to put the model in "training mode", which turns on _dropout_ and _batch normalization_. Although we aren't using them in this model, it's good practice to include it.
# 
# For each batch, we first zero the gradients. Each parameter in a model has a `grad` attribute which stores the gradient calculated by the `criterion`. PyTorch does not automatically remove (or zero) the gradients calculated from the last gradient calculation so they must be manually cleared.
# 
# We then feed the batch of sentences, `batch.text`, into the model. Note, you do not need to do `model.forward(batch.text)`, simply calling the model works. The `squeeze` is needed as the predictions are initially size _**[batch size, 1]**_, and we need to remove the dimension of size 1.
# 
# The loss and accuracy are then calculated using our predictions and the labels, `batch.label`. 
# 
# We calculate the gradient of each parameter with `loss.backward()`, and then update the parameters using the gradients and optimizer algorithm with `optimizer.step()`.
# 
# The loss and accuracy is accumulated across the epoch, the `.item()` method is used to extract a scalar from a tensor which only contains a single value.
# 
# Finally, we return the loss and accuracy, averaged across the epoch. The len of an iterator is the number of batches in the iterator.
# 
# You may recall when initializing the `LABEL` field, we set `tensor_type=torch.FloatTensor`. This is because TorchText sets tensors to be `LongTensor`s by default, however our criterion expects both inputs to be `FloatTensor`s. As we have manually set the `tensor_type` to be `FloatTensor`s, this conversion is done for us.
# 
# Another method would be to do the conversion inside the `train` function by passing `batch.label.float()` instad of `batch.label` to the criterion. 

# In[21]:


def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        predictions = model(batch.text).squeeze(1)
        
        loss = criterion(predictions, batch.label)
        
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# `evaluate` is similar to `train`, with a few modifications as you don't want to update the parameters when evaluating.
# 
# `model.eval()` puts the model in "evaluation mode", this turns off _dropout_ and _batch normalization_. Again, we are not using them in this model, but it is good practice to include it.
# 
# Inside the `no_grad()`, no gradients are calculated which speeds up computation.
# 
# The rest of the function is the same as `train`, with the removal of `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`.

# In[22]:


def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# We then train the model through multiple epochs, an epoch being a complete pass through all examples in the split.

# In[23]:


N_EPOCHS = 5

for epoch in range(N_EPOCHS):

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    print('Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc*100:.2f}%, Val. Loss: {valid_loss:.3f}, Val. Acc: {valid_acc*100:.2f}%')


# You may have noticed the loss is not really decreasing and the accuracy is poor. This is due to several issues with the model which we'll improve in the next notebook.
# 
# Finally, the metric you actually care about, the test loss and accuracy.

# In[24]:


test_loss, test_acc = evaluate(model, test_iterator, criterion)

print('Test Loss: {test_loss:.3f}, Test Acc: {test_acc*100:.2f}%')


# ## Next Steps
# 
# In the next notebook, the improvements we will make are:
# - different optimizer
# - use pre-trained word embeddings
# - different RNN architecture
# - bidirectional RNN
# - multi-layer RNN
# - regularization
# 
# This will allow us to achieve ~85% accuracy.
