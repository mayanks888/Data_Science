# import tensorflow.keras.models.Sequential  as tf
from keras.layers import Convolution2D
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
# tf.keras.models.Sequential
#initialise sequential layer of cnn
myclassifier=Sequential()
# step 1: convolution layer

myclassifier.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1),padding="Same",activation='relu',input_shape=(64,64,3)))
myclassifier.add(MaxPooling2D(pool_size=(2,2)))

myclassifier.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="Same",activation='relu'))
myclassifier.add(MaxPooling2D(pool_size=(2,2)))
#convert matrix in single row matrix
myclassifier.add(Flatten())

#fully connected layer
myclassifier.add(Dense(output_dim=128, activation="relu"))

# output layer?
myclassifier.add(Dense(output_dim=1,activation='sigmoid'))

myclassifier.compile(optimizer="adam",loss='binary_crossentropy',metrics=['accuracy'])



#image argumentation
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_data = train_datagen.flow_from_directory(directory="C:/Users/mayank/Documents/Datasets/Cat_dogs/train",
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')
# training_data = train_datagen.flow_from_directory(directory="../../Datasets/Cat_dogs/train/",
#                                                     target_size=(64, 64),
#                                                     batch_size=32,
#                                                     class_mode='binary')

test_validation = test_datagen.flow_from_directory("C:/Users/mayank/Documents/Datasets/Cat_dogs/test",
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='binary')

myclassifier.fit_generator(generator=training_data,
                            steps_per_epoch=1,
                            epochs=1,
                            validation_data=test_validation,
                            validation_steps=200)

# making new prediction

import numpy as np
from keras.preprocessing import image

test_image=image.load_img(path="C:/Users/mayank/Documents/Datasets/Cat_dogs/test1/509.jpg",target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=myclassifier.predict(test_image)
training_data.class_indices
print((int(result)))



