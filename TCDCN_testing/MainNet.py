import load_3 as ld
import tensorflow as tf
import numpy as np
from scipy import linalg as LG
import time
"""
    Memory control code
"""

config_proto = tf.ConfigProto()
config_proto.gpu_options.allow_growth = True
config_proto.allow_soft_placement = True
config_proto.log_device_placement = False
class TCDCN:
	# Create weight variables for a given layer
    	def weight_variable(shape,name="generic"):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)
    	def bias_variable(shape,name="generic"):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)
    	def conv2d(x, W,name="generic"):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
    	def max_pool_2x2(x,name="generic"):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='VALID')
    	def dynamic_task_coefficient(aux_task_num):
		# init mean to zero
            initial = tf.truncated_normal([aux_task_num,], stddev=1)
            return tf.Variable(initial)
    	def __init__(self,logs_path="/home/ekram67/Projects/face_features/log_files"):
		self.covariance_matrix = tf.placeholder(tf.float32,shape=[21,21])
		self.name ="Task Constrained Deep Convolutional Networks"
        	self.logs_path = logs_path
        	self.train_writer = tf.summary.FileWriter(self.logs_path)
		# Kernel shapes - Main task
		self.kernel_1_shape = [5, 5, 3, 20]
		self.kernel_2_shape = [5,5,20,48]
		self.kernel_3_shape = [3,3,48,64]
		self.kernel_4_shape = [3,3,64,80]
		self.fc_weights_training = [3920, 80]
		self.fc_weights_testing = [13520, 80]
		# Bias shapes - Main task
		self.bias_shape_1 = [20]
		self.bias_shape_2 = [48]
		self.bias_shape_3 = [64]
		self.bias_shape_4 = [80]
		self.fc_bias = [80]
		# Weights - auxillary task
		self.gender_shape = [80,2]
		self.smile_shape = [80,2]
		self.glasses_shape = [80,2]
		self.landmark_shape = [80,10]
		self.headpose_shape = [80,5]
		# Bias shape - auxillary task
		self.gender_bias = [2]
		self.smile_bias = [2]
		self.glasses_bias = [2]
		self.landmark_bias = [10]
		self.headpose_bias = [5]
		# dynamic task coefficients vector
                self.dtc_vec =  tf.Variable(tf.truncated_normal([4], stddev=1))
                # Convolutional layers - Main Task
                self.W_conv1 = tf.Variable(tf.truncated_normal(self.kernel_1_shape, stddev=0.1))
                self.W_conv2 = tf.Variable(tf.truncated_normal(self.kernel_2_shape, stddev=0.1))
                self.W_conv3 = tf.Variable(tf.truncated_normal(self.kernel_3_shape, stddev=0.1))
                self.W_conv4 = tf.Variable(tf.truncated_normal(self.kernel_4_shape, stddev=0.1))
                self.bias_main_1 = tf.Variable(tf.constant(0.1, shape=self.bias_shape_1))
                self.bias_main_2 = tf.Variable(tf.constant(0.1, shape=self.bias_shape_2))
                self.bias_main_3 = tf.Variable(tf.constant(0.1, shape=self.bias_shape_3))
                self.bias_main_4 = tf.Variable(tf.constant(0.1, shape=self.bias_shape_4))
                # Fully connected layer - Main task
                self.fc_layer_training = tf.Variable(tf.truncated_normal(self.fc_weights_training, stddev=0.1))
                self.fc_layer_testing = tf.Variable(tf.truncated_normal(self.fc_weights_testing, stddev=0.1))
		self.fc_bias = tf.Variable(tf.constant(0.1, shape=self.fc_bias))
                # auxiallary task
                self.landmark_weights = tf.Variable(tf.truncated_normal(self.landmark_shape,stddev=0.1))
		self.landmark_bias_aux = tf.Variable(tf.constant(0.1, shape=self.landmark_bias))
                self.gender_weights = tf.Variable(tf.truncated_normal(self.gender_shape, stddev=0.1))
                self.gender_bias_aux = tf.Variable(tf.constant(0.1, shape=self.gender_bias))
                self.smile_weights = tf.Variable(tf.truncated_normal(self.smile_shape, stddev=0.1))
                self.smile_bias_aux = tf.Variable(tf.constant(0.1, shape=self.smile_bias))
                self.glasses_weights = tf.Variable(tf.truncated_normal(self.glasses_shape, stddev=0.1))
                self.glasses_bias_aux = tf.Variable(tf.constant(0.1, shape=self.glasses_bias))
                self.headpose_weights = tf.Variable(tf.truncated_normal(self.headpose_shape, stddev=0.1))
                self.headpose_bias_aux = tf.Variable(tf.constant(0.1, shape=self.headpose_bias))
                # List to keep track of energy functions at different iterations - for part 3
                self.E_list_gender = [1.,1.,1.,1.,1.]
                self.E_list_smile = [1.,1.,1.,1.,1.]
                self.E_list_glasses = [1.,1.,1.,1.,1.]
                self.E_list_headpose = [1.,1.,1.,1.,1.]
                # Part 3 parameters
                self.tau = 5
                self.ro = 5
	def weight_matrix_create(self):
		self.W = tf.concat([self.landmark_weights,self.gender_weights,self.smile_weights,self.glasses_weights,self.headpose_weights],1)
		return self.W

	def build_and_error(self,image,keep_prob,landmark,gender,smile,glasses,headpose,dataset):
		# layer 1
                h_conv1 = tf.abs(tf.nn.tanh(tf.nn.conv2d(image, self.W_conv1, strides=[1, 1, 1, 1], padding='VALID') + self.bias_main_1))
                h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='VALID')
                # layer 2
                h_conv2 = tf.abs(tf.nn.tanh(tf.nn.conv2d(h_pool1, self.W_conv2, strides=[1, 1, 1, 1], padding='VALID') + self.bias_main_2))
                h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='VALID')
                # layer3
                h_conv3 = tf.abs(tf.nn.tanh(tf.nn.conv2d(h_pool2, self.W_conv3, strides=[1, 1, 1, 1], padding='VALID') + self.bias_main_3))
                h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='VALID')
                # layer4
                h_conv4 = tf.abs(tf.nn.tanh(tf.nn.conv2d(h_pool3, self.W_conv4, strides=[1, 1, 1, 1], padding='VALID') + self.bias_main_4))
                h_pool4 = tf.nn.max_pool(h_conv4, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='VALID')
                # layer5
		if dataset == "AFLW":
                        h_pool4_flat = tf.reshape(h_pool4, self.fc_weights_training)
                        h_fc1 = tf.abs(tf.nn.tanh(tf.matmul(tf.transpose(h_pool4_flat), self.fc_layer_training) + self.fc_bias))
                elif dataset == "lfw_5590":
                        h_pool4_flat = tf.reshape(h_pool4, self.fc_weights_testing)
                        h_fc1 = tf.abs(tf.nn.tanh(tf.matmul(tf.transpose(h_pool4_flat), self.fc_layer_testing) + self.fc_bias))
                h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
                # landmark
                y_landmark = tf.matmul(h_fc1_drop, self.landmark_weights) + self.landmark_bias_aux
                # gender
		y_gender = tf.matmul(h_fc1_drop, self.gender_weights) + self.gender_bias_aux
                # smile
                y_smile = tf.matmul(h_fc1_drop, self.smile_weights) + self.smile_bias_aux
                # glasses
                y_glasses = tf.matmul(h_fc1_drop, self.glasses_weights) + self.glasses_bias_aux
                # headpose
                y_headpose = tf.matmul(h_fc1_drop, self.headpose_weights) + self.headpose_bias_aux
                self.y_landmark = y_landmark
                self.y_gender = y_gender
                self.y_smile = y_smile
                self.y_glasses  = y_glasses
                self.y_headpose = y_headpose
		mean_squared_error = tf.losses.mean_squared_error(landmark,y_landmark)
		self.mean_squared_error = mean_squared_error
		loss_ = landmark-self.y_landmark
		loss_square = tf.square(loss_)
		loss_square_reduce = tf.reduce_sum(loss_square)
		softmax_entropy_loss_gender = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_gender,
                                                                                        labels=tf.scalar_mul(self.dtc_vec[0],gender)))
		softmax_entropy_loss_smile = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_smile,
                                                                                          labels=tf.scalar_mul(self.dtc_vec[1],smile)))
		softmax_entropy_loss_glasses = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_glasses,
                                                                                        labels=tf.scalar_mul(self.dtc_vec[2],glasses)))
		softmax_entropy_loss_headpose = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_headpose,
                                                                                    labels=tf.scalar_mul(self.dtc_vec[3],headpose)))
		square_k1 = tf.square(self.W_conv1)
		square_k1_reduce_sum = tf.reduce_sum(square_k1,[0,1,2,3])
		kernel_decay_1 = tf.reduce_sum(tf.square(self.W_conv1))
		kernel_decay_2 = tf.reduce_sum(tf.square(self.W_conv2))
		kernel_decay_3 = tf.reduce_sum(tf.square(self.W_conv3))
		kernel_decay_4 = tf.reduce_sum(tf.square(self.W_conv4))
		weight_decay = tf.trace(tf.matmul(tf.matmul(self.W,self.covariance_matrix),tf.transpose(self.W)))
		# Training algorithm PART 1
        	error_cached = loss_square_reduce + softmax_entropy_loss_gender + softmax_entropy_loss_smile + softmax_entropy_loss_glasses  +\
        	softmax_entropy_loss_headpose + kernel_decay_1 + kernel_decay_2 + kernel_decay_3 + kernel_decay_4 + weight_decay
		self.error_cached = error_cached
		self.loss_square_reduce = loss_square_reduce
		self.softmax_entropy_loss_gender = softmax_entropy_loss_gender
		self.softmax_entropy_loss_glasses = softmax_entropy_loss_glasses
		self.softmax_entropy_loss_headpose = softmax_entropy_loss_headpose
		self.softmax_entropy_loss_smile = softmax_entropy_loss_smile
		# Training algorithm Part 3
		mu_gender = self.ro*((self.E_list_gender[0] - self.E_list_gender[4])/self.E_list_gender[0])
        	mu_glasses = self.ro*((self.E_list_glasses[0] - self.E_list_glasses[4])/self.E_list_glasses[0])
        	mu_smile = self.ro*((self.E_list_smile[0] - self.E_list_smile[4])/self.E_list_smile[0])
        	mu_headpose = self.ro*((self.E_list_headpose[0] - self.E_list_headpose[4])/self.E_list_headpose[0])
        	dynamic_error_cached = -(1/80)*(softmax_entropy_loss_gender +\
                                softmax_entropy_loss_glasses +\
                                softmax_entropy_loss_headpose +\
                                softmax_entropy_loss_smile) +\
                                0.5*((self.dtc_vec[0] - mu_gender) +\
                                (self.dtc_vec[1] - mu_glasses) +\
                                (self.dtc_vec[2] - mu_smile) +\
                                (self.dtc_vec[3] - mu_headpose))
        	self.dynamic_error_cached = dynamic_error_cached
        	self.mu_gender = mu_gender
        	self.mu_glasses = mu_glasses
        	self.mu_smile = mu_smile
         	self.mu_headpose = mu_headpose
        	return error_cached,dynamic_error_cached,softmax_entropy_loss_gender,softmax_entropy_loss_glasses,softmax_entropy_loss_headpose,softmax_entropy_loss_smile

"""
	Main Code
"""
<<<<<<< HEAD
=======
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
        tb_numpy = sess.run(target_batch)
        db_numpy = sess.run(data_batch)
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


# Set up attributes
coord = tf.train.Coordinator()
# Training set
dataset_train = ld.AFLW_Dataset('training.txt',"AFLW")
# Testing set
dataset_test = ld.AFLW_Dataset('testing.txt',"lfw_5590")
# Network
net = TCDCN()
# Training dataset queue
dataset_queue_train = ld.Data_Queue_tf(dataset_train,imagesize=150,capacity=1000)
# Testing dataset queue
dataset_queue_test = ld.Data_Queue_tf(dataset_test,imagesize=250,capacity=1000)
# Get ground truth labels
img_batch_train,landmark_batch_train,gender_batch_train,smile_batch_train,glasses_batch_train,headpose_batch_train = dataset_queue_train.get_input()
img_batch_test,landmark_batch_test,gender_batch_test,smile_batch_test,glasses_batch_test,headpose_batch_test = dataset_queue_test.get_input()
# Create weight matrix
weight_matrix_op = net.weight_matrix_create()
error_op_train = net.build_and_error(img_batch_train,0.5,landmark_batch_train,gender_batch_train,smile_batch_train,
									glasses_batch_train,headpose_batch_train,"AFLW")
error_op_test = net.build_and_error(img_batch_test,0.5,landmark_batch_test,gender_batch_test,smile_batch_test,
                                                                        glasses_batch_test,headpose_batch_test,"lfw_5590")
# Session
with tf.Session(config=config_proto) as sess:
    i = 0
    tf.train.start_queue_runners(sess=sess, coord=coord)
    dataset_queue_test.start_threads(coord,sess,num_threads=13)
    dataset_queue_train.start_threads(coord,sess,num_threads=13)
    while i <= 0:
	print("Running Training set")
    	net.train_writer.add_graph(sess.graph)
    	train_step_1 = tf.train.AdamOptimizer(1e-3).minimize(net.error_cached)
    	train_step_2 = tf.train.AdamOptimizer(1e-3).minimize(net.dynamic_error_cached)
    	summary_op_1 = tf.summary.scalar("Error_Tree_train",net.error_cached)
   	summary_op_2 = tf.summary.scalar("Dynamic_Error_Tree_train",net.dynamic_error_cached)
    	summary_op_3 = tf.summary.scalar("L2 Norm - landmark loss_train",net.loss_square_reduce)
    	summary_op_4 = tf.summary.scalar("Softmax Gender_train",net.softmax_entropy_loss_gender)
    	summary_op_5 = tf.summary.scalar("Softmax Smile_train",net.softmax_entropy_loss_smile)
    	summary_op_6 = tf.summary.scalar("Softmax Glasses_train",net.softmax_entropy_loss_glasses)
    	summary_op_7 = tf.summary.scalar("Softmax Headpose_train",net.softmax_entropy_loss_headpose)
    	summary_op_8 = tf.summary.scalar("Landmark Mean Squared Error_train",net.mean_squared_error)
    	summary_op_merge = tf.summary.merge_all()
	sess.run(tf.global_variables_initializer())
    	# image and feature list represented as tensors
    	for x in range(3000):
        	#Training algorithm PART 2
        	WM_list = sess.run(weight_matrix_op)
        	WM = np.array(WM_list)
        	covariance_num = LG.sqrtm(np.transpose(WM.reshape(80,21)).dot(WM.reshape(80,21)))
        	covariance_den = np.trace(covariance_num)
        	covariance_matrix_ = np.divide(covariance_num,covariance_den)
        	identity_matrix = (10**-8)*(np.identity(21))
        	covariance_matrix_ = identity_matrix + covariance_matrix_
        	covariance_matrix_inverse = LG.inv(covariance_matrix_)
    		L = sess.run([error_op_train,train_step_1,train_step_2,summary_op_merge],
						feed_dict={net.covariance_matrix:covariance_matrix_inverse})
		err,d_err,selg,selgl,selhp,sels = L[0]
    		summ = L[3]
        	net.train_writer.add_summary(summ,x)
    		print("The size of the queue is {}".format(dataset_queue_train.queue.size().eval(session=sess)))
        	print("Loss at Training step {} is {}".format(x,err))
        	print("Dynamic Loss at Training step {} is {}".format(x,d_err))
        	net.E_list_gender.append(selg)
        	if len(net.E_list_gender) > net.tau:
            		net.E_list_gender.pop(0)
        	net.E_list_smile.append(sels)
        	if len(net.E_list_smile) > net.tau:
            		net.E_list_smile.pop(0)
        	net.E_list_glasses.append(selgl)
        	if len(net.E_list_glasses) > net.tau:
            		net.E_list_glasses.pop(0)
        	net.E_list_headpose.append(selhp)
        	if len(net.E_list_headpose) > net.tau:
            		net.E_list_headpose.pop(0)
	i =  1
    print("Running Testing Set")
    net.E_list_gender = [1.,1.,1.,1.,1.]
    net.E_list_smile = [1.,1.,1.,1.,1.]
    net.E_list_glasses = [1.,1.,1.,1.,1.]
    net.E_list_headpose = [1.,1.,1.,1.,1.]
    summary_op_1 = tf.summary.scalar("Error_Tree_test",net.error_cached)
    summary_op_2 = tf.summary.scalar("Dynamic_Error_Tree_test",net.dynamic_error_cached)
    summary_op_3 = tf.summary.scalar("L2 Norm - landmark loss_test",net.loss_square_reduce)
    summary_op_4 = tf.summary.scalar("Softmax Gender_test",net.softmax_entropy_loss_gender)
    summary_op_5 = tf.summary.scalar("Softmax Smile_test",net.softmax_entropy_loss_smile)
    summary_op_6 = tf.summary.scalar("Softmax Glasses_test",net.softmax_entropy_loss_glasses)
    summary_op_7 = tf.summary.scalar("Softmax Headpose_test",net.softmax_entropy_loss_headpose)
    summary_op_8 = tf.summary.scalar("Landmark Mean Squared Error_test",net.mean_squared_error)
    summary_op_merge = tf.summary.merge_all()
    # image and feature list represented as tensors
    for x in range(3000):
    	#Training algorithm PART 2
    	WM_list = sess.run(weight_matrix_op)
        WM = np.array(WM_list)
        covariance_num = LG.sqrtm(np.transpose(WM.reshape(80,21)).dot(WM.reshape(80,21)))
        covariance_den = np.trace(covariance_num)
        covariance_matrix_ = np.divide(covariance_num,covariance_den)
        identity_matrix = (10**-8)*(np.identity(21))
        covariance_matrix_ = identity_matrix + covariance_matrix_
        covariance_matrix_inverse = LG.inv(covariance_matrix_)
        L = sess.run([error_op_train,summary_op_merge],feed_dict={net.covariance_matrix:covariance_matrix_inverse})
        err,d_err,selg,selgl,selhp,sels = L[0]
        summ = L[1]
        net.train_writer.add_summary(summ,x)
        print("The size of the queue is {}".format(dataset_queue_test.queue.size().eval(session=sess)))
        print("Loss at Testing step {} is {}".format(x,err))
        print("Dynamic Loss at Testing step {} is {}".format(x,d_err))
        net.E_list_gender.append(selg)
        if len(net.E_list_gender) > net.tau:
        	net.E_list_gender.pop(0)
        net.E_list_smile.append(sels)
        if len(net.E_list_smile) > net.tau:
                net.E_list_smile.pop(0)
        net.E_list_glasses.append(selgl)
        if len(net.E_list_glasses) > net.tau:
                net.E_list_glasses.pop(0)
        net.E_list_headpose.append(selhp)
        if len(net.E_list_headpose) > net.tau:
                net.E_list_headpose.pop(0)

