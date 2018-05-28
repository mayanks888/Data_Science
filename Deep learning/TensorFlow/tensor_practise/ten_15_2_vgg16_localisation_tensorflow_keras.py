import cv2
import os
import xml.etree.ElementTree as ET
import  pandas as pd
import numpy as np
from keras.utils import np_utils
from keras import applications
from keras.callbacks import TensorBoard
from tensorflow.python import debug as tf_debug
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
from batchup import data_source

num_classes=4

# dataset = pd.read_csv("SortedXmlresult.csv")
#
# loading my prepared datasets
# dataset = pd.read_csv("/home/mayank-s/PycharmProjects//Datasets/SortedXmlresult_linux.csv")
dataset = pd.read_csv("../../../../Datasets/SortedXmlresult_linux.csv")
x = dataset.iloc[:, 1].values
y = dataset.iloc[:, 2].values
y=dataset.iloc[:, 3:8].values

# new_val_y=np.resize(y,(y.shape[0],1))

#y=y.resize(y.shape[0],1)
# this was used to categorise label if they are more than tow
# y_test = np_utils.to_categorical(y, 2)#cotegorise label

imagelist=[]
for loop in x:
    my_image=cv2.imread(loop,1)#reading my path of all address
    image_scale=cv2.resize(my_image,dsize=(224,224),interpolation=cv2.INTER_NEAREST)#resisizing as per the vgg16 module
    imagelist.append(image_scale)#added all pixel values in list

image_list_array=np.array(imagelist)#convet list into array since all calculation required array input

new_image_input,y=shuffle(image_list_array,y,random_state=4)#shuffle data (good practise)

X_train, X_test, y_train, y_test = train_test_split(new_image_input, y, test_size = .20, random_state = 4)#splitting data (no need if test data is present

# ### Placeholders
input_x = tf.placeholder(tf.float32,shape=[None,224,224,3])
y_true = tf.placeholder(tf.float32,shape=[None,1])
# ### Layers
# x_image = tf.reshape(x,[-1,224,224,3])

# importing vgg base_model
# imput_shape=new_image_input[0].shape#good thing to know the shape of input array
x_image = tf.reshape(input_x,[-1,224,224,3])
# input_tensor = tf.keras.Input(shape=(224, 224, 3))
#
# inputshape=(224,224,3)
base_model=tf.keras.applications.vgg16.VGG16(weights = "imagenet", include_top=True,input_tensor=x_image)#loading vgg16  model trained n imagenet datasets

last_layer = base_model.get_layer('block5_pool').output#taking the previous layers from vgg16 officaal model
x= tf.keras.layers.Flatten(name='flatten')(last_layer)
x = tf.keras.layers.Dense(128, activation='relu', name='fc1')(x)

hold_prob = tf.placeholder(tf.float32)#this made to pass the user defined value for dropout probabilty you could have also used contant value
full_one_dropout = tf.nn.dropout(x,keep_prob=hold_prob)

x = tf.keras.layers.Dense(128, activation='relu', name='fc2')(x)
last_layer_out = tf.keras.layers.Dense(num_classes,name='output')(x)
# custom_vgg_model2 = base_model(input_shape,out)
custom_vgg_model2 = tf.keras.Model(x_image,last_layer_out)

# freeze all the layers except the dense layers
for layer in custom_vgg_model2.layers[:-3]:
	layer.trainable = False

print (custom_vgg_model2.summary())

out=custom_vgg_model2.output

# ---------------------------------------------------------------------------
def find_iou(groundbb, predicted_bb):
    data_ground = groundbb
    data_predicted = predicted_bb
    xminofmax = np.maximum(data_ground[0], data_predicted[0])
    yminofmax = np.maximum(data_ground[1], data_predicted[1])
    xmaxofmin = np.minimum(data_ground[2], data_predicted[2])
    ymaxofmin = np.minimum(data_ground[3], data_predicted[3])

    interction = ((xmaxofmin - xminofmax + 1) * (ymaxofmin - yminofmax + 1))
#   i have added 1 to all the equation save the equation from giving 0 iou value
#    AOG: area of ground box
    AOG = (np.abs(data_ground[0] - data_ground[2]) + 1) * (np.abs(data_ground[1] - data_ground[3]) + 1)
    #AOP:area of predicted box
    AOP = (np.abs(data_predicted[0] - data_predicted[2]) + 1) * (np.abs(data_predicted[1] - data_predicted[3]) + 1)
    union= (AOG + AOP) - interction
    iou = (interction /union)
    mean_iou = np.mean(iou)
    return (iou, mean_iou)


def bbox_overlap_iou(bboxes1, bboxes2):
    """
    Args:
        bboxes1: shape (total_bboxes1, 4)
            with x1, y1, x2, y2 point order.
        bboxes2: shape (total_bboxes2, 4)
            with x1, y1, x2, y2 point order.

        p1 *-----
           |     |
           |_____* p2

    Returns:
        Tensor with shape (total_bboxes1, total_bboxes2)
        with the IoU (intersection over union) of bboxes1[i] and bboxes2[j]
        in [i, j].
    """

    x11, y11, x12, y12 = tf.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = tf.split(bboxes2, 4, axis=1)

    xI1 = tf.maximum(x11, tf.transpose(x21))
    yI1 = tf.maximum(y11, tf.transpose(y21))

    xI2 = tf.minimum(x12, tf.transpose(x22))
    yI2 = tf.minimum(y12, tf.transpose(y22))

    inter_area = (xI2 - xI1 + 1) * (yI2 - yI1 + 1)

    bboxes1_area = (x12 - x11 + 1) * (y12 - y11 + 1)
    bboxes2_area = (x22 - x21 + 1) * (y22 - y21 + 1)

     # inter_area / ((bboxes1_area + tf.transpose(bboxes2_area)) - inter_area)
    return  tf.reduce_mean(inter_area / ((bboxes1_area + tf.transpose(bboxes2_area)) - inter_area))

loss=(bbox_overlap_iou(y_true,out))


# loss=tf.metrics.mean_iou(labels=y_true,predictions=out,num_classes=4)
# -------------------------------------------------------------------------------


optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.0001)#Inistising your optimising functions
train = optimizer.minimize(loss)#

correct_prediction = tf.equal(tf.round(out),y_true)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#acc=tf.metrics.accuracy(labels=y_true,predictions=out)
# intialiasing all variable

init=tf.global_variables_initializer()

epochs=20

sess=tf.Session()
sess.run(init)
# sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:7000")
# X_train, X_test, y_train, y_test = train_test_split(new_image_input, y, test_size = .10, random_state = 4)#splitting data (no need if test data is present

for loop in range(epochs):
    # ______________________________________________________________________
    # my batch creater
    # Construct an array data source
    ds = data_source.ArrayDataSource([X_train, y_train])
    # Iterate over samples, drawing batches of 64 elements in random order
    for (batch_X, batch_y) in ds.batch_iterator(batch_size=10, shuffle=True):#shuffle true will randomise every batch
        _,accuracy=sess.run([train,acc], feed_dict={input_x: batch_X, y_true: batch_y, hold_prob: 0.5})
        print("the train accuracy is:", accuracy)
         # ________________________________________________________________________

    if loop % 1 == 0:
        print('Currently on step {}'.format(loop))
        print('Accuracy is:')
        # Test the Train Modelsess.run(train, feed_dict={x: batch_x, y_true: batch_y, hold_prob: 0.5})
        correct_prediction = tf.equal(tf.round(out), y_true)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(acc, feed_dict={x: X_test, y_true: y_test, hold_prob: 1.0}))

