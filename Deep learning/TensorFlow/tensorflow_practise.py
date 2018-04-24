import tensorflow as tf

x = tf.placeholder("float", [None, 3])
y=tf.Variable(initial_value=[[2,2,2],
                            [2,2,2],])

y = x * 2
z = y + 3
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

x_data = [[1, 2, 3],
         [4, 5, 6],]
result = sess.run(z, feed_dict={x: x_data})
# new_result=sess.run(y,feed_dict={x: x_data})
# print(sess.run(x))

print(result)
# print (new_result)