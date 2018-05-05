from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import math
import numpy as np
import tensorflow as tf

new_data=np.arange(0,10,2)
print(new_data)
data=make_blobs(50,2,2,random_state=75)
print(data)
feature=data[0]
labels=data[1]
x=new_data
y=(-.923*x)/.60083
plt.plot(x,y)
plt.scatter(feature[:,0],feature[:,1],c=labels)
plt.show()
wih=np.random.rand(3,3)-.5
wih = np.random.normal(0, pow(1, -0.5), (2,3))

print (tf.__version__)#check tensor flow version
