import cv2
import numpy as np
import tensorflow as tf
import sklearn.preprocessing
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.utils import np_utils
from batchup import data_source
from tensorflow.python import debug as tf_debug


# linux

# data=pd.read_csv('../../../../Datasets/MNIST_data/test_image (copy).csv')
# label=pd.read_csv('../../../../Datasets/MNIST_data/test_label (copy).csv')

data=pd.read_csv('../../../../Datasets/MNIST_data/train_image.csv')
label=pd.read_csv('../../../../Datasets/MNIST_data/train_label.csv')

test_feature=pd.read_csv('../../../../Datasets/MNIST_data/test_image.csv')
test_label=pd.read_csv('../../../../Datasets/MNIST_data/test_label.csv')

#winodws

# data=pd.read_csv(r"C:\Users\mayank\Documents\Datasets\MNIST_data\test_image.csv")
# label=pd.read_csv(r"C:\Users\mayank\Documents\Datasets\MNIST_data\test_label.csv")
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


feature_input = data.iloc[:,:].values
y = label.iloc[:,:].values
_______________________________

scaled_input = np.asfarray(feature_input/255.0)# * 0.99) +0.01

# '_---______________________________________________' \
#one hot encode label data
y_train = np_utils.to_categorical(y, 10)
# '_---______________________________________________'
# scaling and one hot encode applied on a test datasets
feature_test = test_feature.iloc[:,:].values
label_test = test_label.iloc[:,:].values
scaled_test = np.asfarray(feature_test/255.0)
y_test = np_utils.to_categorical(label_test, 10)
# '_---______________________________________________'

# input_shape = Input(shape=(224, 224, 3))
inputshape=(224,224,3)
base_model=tf.keras.applications.vgg16.VGG16(weights = "imagenet", include_top=True,input_tensor=input_shape)
print (base_model.summary())

num_classes=1
last_layer = base_model.get_layer('block5_pool').output#taking the previous layers from vgg16 officaal model
x= tf.keras.layers.Flatten(name='flatten')(last_layer)
x = tf.keras.layers.Dense(128, activation='relu', name='fc1')(x)
x = tf.keras.layersDense(128, activation='relu', name='fc2')(x)
out = tf.keras.Dense(num_classes, activation='sigmoid', name='output')(x)
# custom_vgg_model2 = base_model(input_shape,out)
custom_vgg_model2 = tf.keras.Model(inputshape,out)


# freeze all the layers except the dense layers
for layer in custom_vgg_model2.layers[:-3]:
	layer.trainable = False

custom_vgg_model2.summary()#