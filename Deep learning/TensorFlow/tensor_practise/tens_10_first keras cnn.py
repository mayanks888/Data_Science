from  keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import  MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

myclassifier=Sequential()
myclassifier.add(Convolution2D())