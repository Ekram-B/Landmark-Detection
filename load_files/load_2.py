"""
Changes
1. Added modules whose services we employ for our own software. 
2. Added code in the 'Main' segment to ensure that code compiles.
3. Added Data_Queue class -- compiled.
4.

"""
# Provides services to interface with an underlying operating system
import os
# Provides functions for duplicating objects using shallow or deep copy semantics.
import copy
# aggreation of unix style path name pattern processing services
import glob
# aggretation of services required for multithreading system
import threading
# aggregation of efficient array operations
import numpy as np
# aggregation of scientific and numerical tools
import scipy.io
import scipy.ndimage as spnd
# aggregation of efficient tensor operations
import tensorflow as tf
# PIL = Python Imaging Library
from PIL import Image


# Global Variables
current_working_directory = os.getcwd()
# Datapath for AFLW code
data_path = "/home/ekram67/Datasets/AFLW"
version_number = ""

# Classes
class AFLW_Dataset:
	# Constructors
	def __init__(self,idx,data_partition):
		# Creating instance variables
    		self.data_path = "/home/ekram67/Datasets"
       		self.version_number = ""
        	# Determining file path of dataset
        	self.dir = os.path.join(data_path,version_number)
        	self.dataset = "AFLW"
		self.name = '{}_{}'.format(self.dataset,self.version_number)
		# get list of all files - use build in open() function rather the PIL routine
		self.all_idxs = open('{}/{}'.format(self.data_path,data_partition)).read().splitlines()
	def num_imgs(self):
		return len(self.all_idxs)
	def load_image(self, idx):
		im = Image.open(os.path.join(self.data_path,self.dataset,'/{}.jpg'.format(idx)))
		im = np.array(im, dtype=np.float32)
		return im

class Data_Queue:
    def __init__(self,dataset,augment=False,patch_size=96,batch_size=5,seed=0,capacity=None,number_of_batches=None):
        # Define queue capacity
        if capacity is None:
            capcity = 10*batch_size
        self.pad_size = patch_size
        self.patch_size = patch_size
        self.dataset = dataset
        print('Dataset {} to queue with {} images'.format(dataset.name.dataset.num_imgs()))
        # Setting up placeholders for tensor flow sessions
        self.queue_input_data = tf.placeholder(tf.float32, shape=[patch_size, patch_size, 3])
        self.queue_input_target = tf.placeholder(tf.int32, shape=[patch_size, patch_size])
        self.queue_input_weights = tf.placeholder(tf.float32, shape=[patch_size, patch_size])
        
        self.queue = tf.FIFOQueue(capacity=capacity,
                                  dtypes=[tf.float32, tf.int32, tf.float32],
                                  shapes=[[patch_size,patch_size,3],
                                          [patch_size,patch_size],
                                          [patch_size,patch_size]])

        self.enqueue_op = self.queue.enqueue([self.queue_input_data,
                                              self.queue_input_target,
                                              self.queue_input_weights])
        self.dequeue_op = self.queue.dequeue()
        self.rnd = np.random.RandomState(seed)
        self.augment = augment
        
        if num_batches is None:
            batch = tf.train.batch(self.dequeue_op, batch_size = batch_size,capacity=capacity,name ='batch_' + dataset.name)
            self.data_batch, self.target_batch, self.weights_batch = batch
        
        else:
            self.data_batch, self.target_batch, self.weights_batch = [], [], []
            for i in range(num_batches):
                batch = tf.train.batch(self.dequeue_op, batch_size=batch_size,
                                       capacity=capacity,
                                       name='batch_'+dataset.name+'%02d'%i)
                self.data_batch.append(batch[0])
                self.target_batch.append(batch[1])
                self.weights_batch.append(batch[2])

        self.cind_lock = threading.Lock() 

# Main
dataset = AFLW_Dataset('1000-image66034','training.txt')
"""
	TODO  - fix this function to provide a number (integer))
"""
print("The number of images to be trained is {}".format(dataset.num_imgs()))
