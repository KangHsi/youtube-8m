import tensorflow as tf
A=tf.ones([6])
B=tf.reshape(A,[-1,2])


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print B.eval() #