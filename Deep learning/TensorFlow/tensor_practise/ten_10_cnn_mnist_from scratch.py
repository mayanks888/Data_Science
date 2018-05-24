import cv2
import numpy as np
import tensorflow as tf
import sklearn.preprocessing
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder


data=pd.read_csv('../../../../Datasets/MNIST_data/test_image.csv')
label=pd.read_csv('../../../../Datasets/MNIST_data/test_label.csv')
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
one_hot_encode=OneHotEncoder()
hot_e=one_hot_encode.fit_transform(y,y=10)
# first_encode=np.reshape(y,newshape=(9,len(y)))
# hot_e=one_hot_encode.fit_transform(first_encode)
print(hot_e)
mydata=hot_e

# '_---______________________________________________'
'''y_test = np_utils.to_categorical(y, 9)#cotegorise label

# new_image_input,y=shuffle(image_list_array,y,random_state=4)#shuffle data (good practise)

X_train, X_test, y_train, y_test = train_test_split(new_image_input, y, test_size = .10, random_state = 4)#splitting data (no need if test data is present
'''
# _______
