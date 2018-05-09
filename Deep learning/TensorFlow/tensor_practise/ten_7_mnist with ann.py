import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data/',one_hot=1)
my_image=mnist.train.images[26].reshape(28,28)#printing any image at index image[index] and shaping it in28X28
plt.imshow(my_image,cmap="gist_gray")
plt.show()

total_pixel_input=728#28X28 matrix
total_neuron=10#this does not have hidded layer as of now so we are considering 10 output sinceoutput can be 0 to 9

#creating placeholder
input_matrix=tf.placeholder(dtype=tf.float32,shape=([None,total_pixel_input]))#(no of rows*728)
output_matrix=tf.placeholder(dtype=tf.float32,shape=([None,total_neuron]))

#creating variable
w=tf.Variable(tf.random_normal(shape=[total_pixel_input,total_neuron]))#(728*10)
b=tf.Variable(tf.ones([10]))
matrix_multiply=tf.matmul(input_matrix,w)+b #output should be a matrix of (10 colum and batch side rows)

#apply softmax  as aactivation function for all ten neuron
#remember here only one softmax is applied to ouput of all 10 neuron
softmax_out=tf.nn.softmax(matrix_multiply)



loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=output_matrix, logits=softmax_out))
    # tf.summary.scalar('cross-entropy', loss)

optimiser=tf.train.GradientDescentOptimizer(learning_rate=.003).minimize(loss)

init=tf.global_variables_initializer()


sess=tf.Session()
sess.run(init)
training_steps=100

for i in range(training_steps):
    batch_x,batch_y=mnist.train.next_batch(100)
    # my_comp_data=sess.run([y,train_data,],feed_dict={input_matrix:batch_x,output_matrix:batch_y})
    mycross_entropy=sess.run(loss,feed_dict={input_matrix:batch_x,output_matrix:batch_y})
