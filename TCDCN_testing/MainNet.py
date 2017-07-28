import load_3 as ld
import tensorflow as tf
import numpy as np
"""
    Generate system of linear equations where for a given tensor - number
    of rows corresponds to the number of neurons.
"""
# Create weight variables for a given layer
def weight_variable(shape):
    """
        Outputs random values from a truncated normal distribution. Note
        that values sampled that are more than 2 normal standard deviations
        away from the mean are dropped and resampled.
    """
    """
        Input: Shape of the kernal matrix - typically 3x3
        Output: Returns a tensor variable wrapping around the kernel
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    """
        Input: Shape of the bias tensor
        Output: Returns a tensor variable wrapping around a tensorflow constant
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x, W):
    """
        Input: X being the image, and W being the kernel weights
        Output: The feature map resultant from the 2D convulution
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
def max_pool_2x2(x):
    """
        Returns a lower dimensionality version of the feature map.
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='VALID')
"""
    Main code
"""
image = tf.placeholder(tf.float32, shape=[150,150,3])
landmark = tf.placeholder(tf.float32, shape=[None, 10])
gender = tf.placeholder(tf.float32, shape=[None, 2])
smile = tf.placeholder(tf.float32, shape=[None, 2])
glasses = tf.placeholder(tf.float32, shape=[None, 2])
headpose = tf.placeholder(tf.float32, shape=[None, 5])
"""
layer 1
"""
x_image = tf.reshape(image, [-1,150,150,3])
W_conv1 = weight_variable([5, 5, 3, 20])
b_conv1 = bias_variable([20])
h_conv1 = tf.abs(tf.nn.tanh(conv2d(x_image, W_conv1) + b_conv1))
h_pool1 = max_pool_2x2(h_conv1)
W_conv2 = weight_variable([3,3,20,48])
b_conv2 = bias_variable([48])
h_conv2 = tf.abs(tf.nn.tanh(conv2d(h_pool1, W_conv2) + b_conv2))
h_pool2 = max_pool_2x2(h_conv2)
# layer3
W_conv3 = weight_variable([3,3,48,50])
b_conv3 = bias_variable([50])
h_conv3 = tf.abs(tf.nn.tanh(conv2d(h_pool2, W_conv3) + b_conv3))
h_pool3 = max_pool_2x2(h_conv3)
# layer4
W_conv4 = weight_variable([2,2,50,80])
b_conv4 = bias_variable([80])
h_conv4 = tf.abs(tf.nn.tanh(conv2d(h_pool3, W_conv4) + b_conv4))
h_pool4 = h_conv4
#  layer5
W_fc1 = weight_variable([200, 80])
b_fc1 = bias_variable([80])
h_pool4_flat = tf.reshape(h_pool4, [-1, 200])
"""
    Encoding the high level features learned to representation interms of the weights
    of the fully connected layers
"""
h_fc1 = tf.abs(tf.nn.tanh(tf.matmul(h_pool4_flat, W_fc1) + b_fc1))
keep_prob = tf.placeholder(tf.float32)
"""
    Regularization technique to reduce overfitting = Dropout
"""
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# readout layer
# landmark
W_fc_landmark = weight_variable([80, 10])
b_fc_landmark = bias_variable([10])
y_landmark = tf.matmul(h_fc1_drop, W_fc_landmark) + b_fc_landmark
# gender
W_fc_gender = weight_variable([80, 2])
b_fc_gender = bias_variable([2])
y_gender = tf.matmul(h_fc1_drop, W_fc_gender) + b_fc_gender
# smile
W_fc_smile = weight_variable([80, 2])
b_fc_smile = bias_variable([2])
y_smile = tf.matmul(h_fc1_drop, W_fc_smile) + b_fc_smile
# glasses
W_fc_glasses = weight_variable([80, 2])
b_fc_glasses = bias_variable([2])
y_glasses = tf.matmul(h_fc1_drop, W_fc_glasses) + b_fc_glasses
# headpose
W_fc_headpose = weight_variable([80, 5])
b_fc_headpose = bias_variable([5])
y_headpose = tf.matmul(h_fc1_drop, W_fc_headpose) + b_fc_headpose
"""
The 1/2 makesthe derivative of l2_norm loss simpler
"""
error = 1 / 2 * tf.reduce_sum(tf.square(landmark - y_landmark)) + \
        tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y_gender, labels=gender)) + \
        tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y_smile, labels=smile)) + \
        tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y_glasses, labels=glasses)) + \
        tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y_headpose, labels=headpose))+\
        2*tf.nn.l2_loss(W_fc_landmark)+\
        2*tf.nn.l2_loss(W_fc_glasses)+\
        2*tf.nn.l2_loss(W_fc_gender)+\
        2*tf.nn.l2_loss(W_fc_headpose)+\
        2*tf.nn.l2_loss(W_fc_smile)
landmark_error = 1 / 2 * tf.reduce_sum(tf.square(landmark - y_landmark))
# train
train_step = tf.train.AdamOptimizer(1e-3).minimize(error)
# Set up data_queue
coord = tf.train.Coordinator()
dataset = ld.AFLW_Dataset('testing.txt')
with tf.Session() as sess:
    # Define dataset queue
    dataset_queue = ld.Data_Queue(dataset,imagesize=150,capacity=1000)
    # Initialize global variables
    sess.run(tf.global_variables_initializer())
    # Start enqueing thread
    dataset_queue.start_threads(coord,sess,num_threads=1)
    for x in range(200000):
        data_batch,target_batch = dataset_queue.get_input()
        # get numpy versions
	print(target_batch)
	print(data_batch)
	print(target_batch)
	print(data_batch)
        tb_numpy = sess.run(target_batch.eval())
        db_numpy = sess.run(data_batch.eval())
	print(tb_numpy)
        i,j,k,l,m,n = dataset_queue.feature_hot_encoding(dataset.list_to_dict(tb_numpy),img) 
        print(x, sess.run(error,feed_dict={image:i,landmark: j.reshape((-1,10)), gender: k.reshape((-1,2)), smile: l.reshape((-1,2)), glasses: m.reshape((-1,2)), headpose: n.reshape((-1,5)), keep_prob: 1}))
        sess.run(train_step,feed_dict={image:i,landmark: j.reshape((-1,10)), gender: k.reshape((-1,2)), smile: l.reshape((-1,2)), glasses: m.reshape((-1,2)), headpose: n.reshape((-1,5)),keep_prob: 0.5})
        data_batch,target_batch = dataset_queue.get_input()
        # get numpy versions
        tb_numpy = sess.run(target_batch.eval())
        db_numpy = sess.run(data_batch.eval())
        o,p,q,r,s,t = dataset_queue.feature_hot_encoding(dataset.list_to_dict(tb_numpy),img) 
        print("testing", sess.run(error, feed_dict={image:o,landmark: p.reshape((-1,10)), gender: q.reshape((-1,2)), smile: r.reshape((-1,2)), glasses: s.reshape((-1,2)), headpose: t.reshape((-1,5)),keep_prob: 1}))
        print("landmark",sess.run(landmark_error,feed_dict={image:o,landmark: p.reshape((-1,10)), gender: q.reshape((-1,2)), smile: r.reshape((-1,2)), glasses:s.reshape((-1,2)), headpose: t.reshape((-1,5)),keep_prob: 1}))



