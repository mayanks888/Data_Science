import cv2
import numpy as np
import tensorflow as tf
import sklearn.preprocessing
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.utils import to_categorical
from keras.utils import np_utils

# data=pd.read_csv('../../../../Datasets/MNIST_data/test_image.csv')
# label=pd.read_csv('../../../../Datasets/MNIST_data/test_label.csv')
# winodws

data=pd.read_csv(r"C:\Users\mayank\Documents\Datasets\MNIST_data\test_image.csv")
label=pd.read_csv(r"C:\Users\mayank\Documents\Datasets\MNIST_data\test_label.csv")
# print (data.head())
# print(label.head())
# '____________________________________________________________'
# to read particular row in datasets
# reading in opencv
'''single_image= data.iloc[0]
single_image_array=np.array(single_image,dtype='uint8')
single_image_array=single_image_array.reshape(28,28)
cv2.imshow("image",single_image_array)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
# '____________________________________________________________'

#dataset = pd.read_csv("SortedXmlresult_linux.csv")
x = data.iloc[:,:].values
y = label.iloc[:,:].values
#to check for any null value in datasets
# print(data.isnull().sum())
# print(label.isnull().sum())
scaled_input = np.asfarray(x/255.0)# * 0.99) +0.01
# this was used to categorise label if they are more than tow
# '_---______________________________________________' \
#one hot encode label data
y_test = np_utils.to_categorical(y, 10)
print(y_test)
# '_---______________________________________________'
# new_image_input,y=shuffle(image_list_array,y,random_state=4)#shuffle data (good practise)

# X_train, X_test, y_train, y_test = train_test_split(new_image_input, y, test_size = .10, random_state = 4)#splitting data (no need if test data is present

def create_weight(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)
#we have to create this many function because of normal and convolution layers
def create_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)

def create_convd(x,W):
    return (tf.nn.conv2d(input=x,filter=W,stride=[1,1,1,1],padding="SAME"))

def max_pool(x):
    return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def convolutional_layer(input_x, shape):
    W = create_weight(shape)
    b = create_bias([shape[3]])
    return tf.nn.relu(create_convd(input_x, W) + b)

def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = create_weight([input_size, size])
    b = create_bias([size])
    return tf.matmul(input_layer, W) + b

# ### Placeholders
x = tf.placeholder(tf.float32,shape=[None,784])
y_true = tf.placeholder(tf.float32,shape=[None,10])
# ### Layers
x_image = tf.reshape(x,[-1,28,28,1])

# create first layer
conv1=convolutional_layer(input_x=x_image,shape=(6,6,1,32))
max_pl1=max_pool(conv1)#max pool will divide the layer into half

# second LAYER
conv2=convolutional_layer(input_x=max_pl1,shape=(6,6,32,64))
max_pl2=max_pool(conv2)


#fully connected laeyr
convo_2_flat = tf.reshape(max_pl2,[-1,7*7*64])#check about this -1
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat,1024))

