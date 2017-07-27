"""
Changes
1. Added modules whose services we employ for our own software.
2. Added code in the 'Main' segment to check and test code.
3. Added Data_Queue class -- compiled.
4. Checked if class was created successfully - yes
5. Checked if images were loaded successfully - yes
6. Adjusted constructors and load services and added a service to print shape.
7. Created dictionary for dataset augmentation - removed rotate pair from dictionary 
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


# Classes
class lfw_5590_Dataset:
    # Constructors
    def __init__(self,data_partition, augment):
        # Creating instance variables
        self.data_path = "/home/ekram67/Datasets"
 	self.version_number = ""
	self.augment = augment
	# Determining file path of dataset
	self.dir = os.path.join(self.data_path,self.version_number)
	self.dataset = "lfw_5590"
	self.name = '{}_{}'.format(self.dataset,self.version_number)
	# get list of all files - use build in open() function rather the PIL routine
	self.all_idxs = open('{}/{}'.format(self.data_path,data_partition)).read().splitlines()

    def num_imgs(self):
        return len(self.all_idxs)
    def load_image(self, idx):
        im = Image.open(os.path.join(self.data_path,self.dataset,'{}.jpg'.format(idx)))
        im = np.array(im, dtype=np.float32)
        return im
    def image_shape(self,idx):
        im = Image.open(os.path.join(self.data_path,self.dataset,'{}.jpg'.format(idx)))
        im = np.array(im, dtype=np.float32)
        print('The image idx is modeled by a ndarray of the shape {}'.format(im.shape))

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

    def enqueue(self,coord,sess):
        # With coord.stop_on_exception():
        while not coord.should_stop():
            self.load_next_batch(sess)

    def start_threads(self,coord,sess,num_threads=1):
        self.cind = 0
        enqueue_threads = [threading.Thread(target=self.enqueue, args=[coord,sess])for _ in range(num_threads)]
        for t in enqueue_threads:
            t.daemon = True
            t.start()
        return enqueue_threads

    def _gen_transform(self,img):
        trans = {}
        # Augmenting operations
        if self.augment:
            medval = np.median(np.mean(img,axis=2))
            trans['flip'] = self.rnd.rand() > 0.5
            trans['contrast'] = self.rnd.lognormal(sigma=0.25)
            trans['brightness'] = np.clip(16*self.rnd.randn(),-medval,255-medval)
            trans['patch_origin'] = np.array([img.shape[i]/2.0 + max(0,img.shape[i] - self.patch_size/max(np.abs(np.cos(trans['rotate'])),np.abs(np.sin(trans['rotate']))))*(self.rnd.rand()-0.5) for i in range(2)])
        else:
            trans['patch_origin'] = np.array([img.shape[i]/2.0 + max(0,img.shape[i] - self.patch_size)*(self.rnd.rand()-0.5) for i in range(2)])
        return trans

    def _apply_transform(self,img,labels,trans=None):
        if trans is None:
            trans = self._gen_transform(img)
        img = np.require(img,dtype=np.float32)
        # Colour augmentations
	if 'brightness' in trans:
            img = np.clip(img + trans['brightness'],0,255)
	if 'contrast' in trans:
            mean_rgb = np.mean(img,axis=(0,1),keepdims=True)
            img = np.clip(trans['contrast']*(img - mean_rgb) + mean_rgb,0,255)

        # Geometric augmentations
        origin = trans['patch_origin']
        if 'flip' in trans:
            img = img[:,:-1,:]
            labels = labels[:,:-1]
            origin[1] = img.shape[1] - origin[1] - 1
        # Finally, extract patch
        ptl = [int(np.round(i - self.patch_size/2)) for i in origin]
        pbr = [i + self.patch_size for i in ptl]
        img_patch = 128 + np.zeros([self.patch_size,self.patch_size,img.shape[2]],dtype=img.dtype)
        real_ptl = [max(0,ind) for ind in ptl]
        real_pbr = [min(img.shape[i],ind) for i,ind in enumerate(pbr)]
        sub_ptl = [ -min(0,ind) for ind in ptl ]
        sub_pbr = [ self.patch_size - max(0,orig-real) for orig,real in zip(pbr,real_pbr) ]
        img_patch   [sub_ptl[0]:sub_pbr[0],sub_ptl[1]:sub_pbr[1],:] = img   [real_ptl[0]:real_pbr[0],real_ptl[1]:real_pbr[1],:]
        return img_patch

    def load_next_batch(self,sess):
	max_ind = self.dataset.num_imgs()

        with self.cind_lock:
            if self.cind == 0:
                self.perm = self.rnd.permutation(self.dataset.num_imgs())
            curr_ind = self.perm[self.cind]
            self.cind = (self.cind+1)%max_ind
	cimg_idx = self.dataset.all_idxs[curr_ind]
        curr_img = self.dataset.load_image(cimg_idx)
        curr_label = self.dataset.load_label(cimg_idx)
        curr_img, curr_label = self._apply_transform(curr_img,curr_label)
        curr_weights = self.dataset.weights[curr_label]
        sess.run(self.enqueue_op, feed_dict={self.queue_input_data: curr_img.astype(np.float32),
                                             self.queue_input_target: curr_label.astype(np.int32),
                                             self.queue_input_weights: curr_weights.astype(np.float32)})

# Main
dataset = lfw_5590_Dataset('training.txt',True)
print("The number of images to be trained is {}".format(dataset.num_imgs()))
dataset.load_image('Aaron_Eckhart_0001')
dataset.image_shape('Aaron_Eckhart_0001')
