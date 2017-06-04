import tensorflow as tf
import tensorflow.contrib.slim as slim
# Create some variables.
with tf.variable_scope("tower"):
  v1 = tf.Variable([10], name="v1",dtype=tf.float32)
with tf.variable_scope("rnn"):
  v1 = tf.Variable([10], name="v2",dtype=tf.float32)
  v2 = tf.Variable([10], name="v2", dtype=tf.float32)
v4=v1+v2
v3=tf.global_variables()

all_vars = tf.trainable_variables()
for i,v in enumerate(all_vars):
    print i
    if v.name[0:3]=='rnn':
      vars1=all_vars[:i]
      vars2=all_vars[i:]
      break

v_vars = [v for v in all_vars if v.name == 'tower/v1:0' or v.name == 'ss/v2:0' or v.name == 'v2:0']
for name in v_vars:
  print name
reshaped_input = slim.batch_norm(
  v2,
  center=True,
  scale=True,
  is_training=True,
  scope="input_bn")

# Add ops to save and restore only 'v2' using the name "my_v2"
saver1 = tf.train.Saver([tower/v1,v2])
print saver1
# {"my_v3": v3}
# Use the saver object normally after that.