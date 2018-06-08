import cv2
import numpy as np
# import tensorflow as tf

import pandas as pd

from keras.utils import np_utils
from batchup import data_source

# from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# ___________________
data=pd.read_csv('../../Datasets/MNIST_data/train_image.csv')
label=pd.read_csv('../../Datasets/MNIST_data/train_label.csv')

test_feature=pd.read_csv('../../Datasets/MNIST_data/test_image.csv')
test_label=pd.read_csv('../../Datasets/MNIST_data/test_label.csv')



# reading in opencv
'''single_image= data.iloc[0]
single_image_array=np.array(single_image,dtype='uint8')
single_image_array=single_image_array.reshape(28,28)
cv2.imshow("image",single_image_array)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
# '____________________________________________________________'

#dataset = pd.read_csv("SortedXmlresult_linux.csv")
feature_input = data.iloc[:,:].values
y = label.iloc[:,:].values

# ________________________________________________________________
# scaling features area image argumentation later we will add more image argumantation function
scaled_input = np.asfarray(feature_input/255.0)# * 0.99) +0.01

# this was used to categorise label if they are more than tow
# '_---______________________________________________' \
#one hot encode label data
y_train = np_utils.to_categorical(y, 10)
# print(y_test)
# '_---______________________________________________'
# new_image_input,y=shuffle(image_list_array,y,random_state=4)#shuffle data (good practise)

# X_train, X_test, y_train, y_test = train_test_split(new_image_input, y, test_size = .10, random_state = 4)#splitting data (no need if test data is present
# '_---______________________________________________'
# scaling and one hot encode applied on a test datasets
feature_test = test_feature.iloc[:,:].values
label_test = test_label.iloc[:,:].values
scaled_test = np.asfarray(feature_test/255.0)
y_test = np_utils.to_categorical(label_test, 10)
# '_---______________________________________________'
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5,padding=1)#this is nothing with(1=channel,layer=32,padding=1 =same size of input)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5,padding=1)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(1600, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = x.float()
        first_layer=self.conv1(x)
        x = F.relu(self.mp(first_layer))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
        # prob=torch.nn.Softmax(x)
        prob=F.log_softmax(x)
        # prob=prob.long()
        return prob


model = Net()
criterea = nn.CrossEntropyLoss()
# criterea=torch.nn.MSELoss(size_average=False)#cross entropy loss
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)




epochs=10
for loop in range(epochs):
    # ______________________________________________________________________
    # my batch creater
    # Construct an array data source
    ds = data_source.ArrayDataSource([scaled_input, y_train])
    # Iterate over samples, drawing batches of 64 elements in random order
    for (data, target) in ds.batch_iterator(batch_size=64, shuffle=True):#shuffle true will randomise every batch
        new_data=data.reshape(-1,1,28,28)
        cool_data=new_data[0]
        # cv2.imshow("image",np.reshape(cool_data,newshape=(28,28,1)))
        # cv2.waitKey(500)
        # cv2.destroyAllWindows()
        # data, target=torch.from_numpy(new_data).double(),torch.from_numpy(target).double()
        data, target = torch.tensor(new_data), torch.tensor(target).long()
        # target=target.long()
        
        # data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target)
        loss = criterea(output[0], target[0])
        loss.backward()
        optimizer.step()
        if loop % 10 == 0:
            print('epoch is', loop)
            print('loss is',loss.data[0])


            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     loop, batch_idx * len(data), len(train_loader.dataset),
            #            100. * batch_idx / len(train_loader), loss.data[0]))

        # _,accuracy=sess.run([train,acc], feed_dict={x: batch_X, y_true: batch_y, hold_prob: 0.5})
        # print("the train accuracy is:", accuracy)
         # ________________________________________________________________________

    # batch_x1, batch_y1 = tf.train.batch([feature_input, y_train], batch_size=50)
    # batch_x,batch_y=sess.run(batch_x1,batch_y1)
    # # sess.run(train, feed_dict={x: batch_x, y_true: batch_y, hold_prob: 0.5})
    #
    # if loop % 1 == 0:
    #     print('Currently on step {}'.format(loop))
    #     print('Accuracy is:')
    #     # Test the Train Modelsess.run(train, feed_dict={x: batch_x, y_true: batch_y, hold_prob: 0.5})
    #     matches = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y_true, 1))
    #
    #     acc = tf.reduce_mean(tf.cast(matches, tf.float32))
    #
    #     print(sess.run(acc, feed_dict={x: scaled_test, y_true: y_test, hold_prob: 1.0}))
    #
    #


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ### Placeholders
"""x = tf.placeholder(tf.float32,shape=[None,784])
y_true = tf.placeholder(tf.float32,shape=[None,10])
# ### Layers
x_image = tf.reshape(x,[-1,28,28,1])

conv_1_layer = tf.keras.layers.Conv2D(filters=32,kernel_size=(6, 6),strides=(1, 1), activation='relu', padding='same', name='block1_conv1')(x_image)
max_pool_1 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv_1_layer)

conv_2_layer = tf.keras.layers.Conv2D(filters=64,kernel_size=(6, 6),strides=(1, 1), activation='relu', padding='same', name='block2_conv1')(max_pool_1)
max_pool_2 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv_2_layer)

flat_layer=tf.keras.layers.Flatten(name='flatten')(max_pool_2)

first_dense=tf.keras.layers.Dense(1024, activation="relu",name="flat1")(flat_layer)

hold_prob = tf.placeholder(tf.float32)#this made to pass the user defined value for dropout probabilty you could have also used contant value
full_one_dropout = tf.nn.dropout(first_dense,keep_prob=hold_prob)

output_layer =tf.keras.layers.Dense(10, activation='relu', name='output1')(full_one_dropout)
# _______________________________________________________

# first method
softmax_output=tf.nn.softmax(logits=output_layer)
entropy_loss_per_row=(y_true * tf.log(softmax_output))#formula for cross entropy
# formula for cross entropy (L=−∑i=0kyilnp^i)
sum_loss_per_row = (-tf.reduce_sum(entropy_loss_per_row,axis=1))#"-" you have to check(axis=1 since I was row wise sum)
# loss_per_row = (-tf.reduce_sum(y_true * tf.log(softmax_output),[1]))#formula for cross entropy
cross_entropy_mean=tf.reduce_mean(sum_loss_per_row)
# -------------------------------------------------------------------------------
# Second method(direct)
# cross_entropy_mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))
# -------------------------------------------------------------------------------
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# optimisation: this is to learning check differnt optimiser
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)#Inistising your optimising functions
train = optimizer.minimize(cross_entropy_mean)#this will trigger backward propogation for learning
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


train_matches = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y_true, 1))
acc = tf.reduce_mean(tf.cast(train_matches, tf.float32))

    # _____


# intialiasing all variable

init=tf.global_variables_initializer()"""