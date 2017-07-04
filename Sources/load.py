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
data_path = "/home/ekram67/Research_Machine_Learning/Source_code/Data_modeling/AFLW" 
version_number = ""

# Classes
class MTFL_Dataset:
	# Constructors
	def __init__(self,idx,dataset, data_path=None,version_number=None):
		# Defensive programming
		if data_path is None:
			self.data_path = "/home/ekram67/Research_Machine_Learning/Source_code/Data_modeling"
		elif version_number is None:
			self.version_number = ""
		# Determining file path of dataset 	
		self.dir = os.path.join(data_path,version_number)
		self.dataset = dataset # testing.txt or training.txt
		self.name = 'MTFL_{}'.format(version_number) 
		self.dataset = dataset		
		# get list of all files - use build in open() function rather the PIL routine
		self.all_idxs = open('{}/Datasets/{}'.format(self.dir,self.dataset)).read().splitlines()
	"""
        Routines
    """
    def num_imgs(self):
		return len(self.all_idxs)

	def load_image(self, idx):
		im = Image.open(os.path.join(self.dir,'/Datasets/','AFLW/{}.jpg'.format(idx)))
		im = np.array(im, dtype=np.float32)
        return im


"""
    Required for handling large datasets
"""
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

def _gen_transform(self,img):
        # Dictionary of transformations to apply to images.
        trans = {}
        if self.augment:
        	# Determine mean of the samples recorded to the vector.
            medval = np.median(np.mean(img,axis=2))
            # Boolean to determine whether to flip the image.
            trans['flip'] = self.rnd.rand() > 0.5
            # Angular distance to rotate image.
            trans['rotate'] = np.pi*(2*self.rnd.rand()-1)
            # Setting contrast to a random variable whose logarithm is normally distributed
            trans['contrast'] = self.rnd.lognormal(sigma=0.25)

            trans['brightness'] = np.clip(16*self.rnd.randn(),-medval,255-medval)
            
            trans['patch_origin'] = np.array([img.shape[i]/2.0 + max(0,img.shape[i] - self.patch_size/max(np.abs(np.cos(trans['rotate'])),np.abs(np.sin(trans['rotate']))))*(self.rnd.rand()-0.5) for i in range(2)])
        else:
            trans['patch_origin'] = np.array([img.shape[i]/2.0 + max(0,img.shape[i] - self.patch_size)*(self.rnd.rand()-0.5) for i in range(2)])
        return trans

def _apply_transform(self,img,labels,trans=None):
        if trans is None:
            trans = _gen_transform(img)
   		# Convert entries to types and sizes of float32         
        img = np.require(img,dtype=np.float32)
        # Colour augmentations

        # Peform brightness augmentations
        if 'brightness' in trans:
            img = np.clip(img + trans['brightness'],0,255)
        # Peform contrast augmentations
        if 'contrast' in trans:
            mean_rgb = np.mean(img,axis=(0,1),keepdims=True)
            img = np.clip(trans['contrast']*(img - mean_rgb) + mean_rgb,0,255)
        
        # Geometric augmentations
        origin = trans['patch_origin'] 
        if 'flip' in trans:
            img = img[:,:-1,:]
            labels = labels[:,:-1]
            origin[1] = img.shape[1] - origin[1] - 1 
        if 'rotate' in trans:
            inimg_shape = img.shape[:2]
            img = spnd.interpolation.rotate(img,(180.0/np.pi)*trans['rotate'],mode='constant',cval=128,order=2,reshape=True)
            labels = spnd.interpolation.rotate(labels,(180.0/np.pi)*trans['rotate'],mode='constant',cval=255,order=0,reshape=True)
            matrix = np.array([[ np.cos(trans['rotate']), -np.sin(trans['rotate'])],
                               [ np.sin(trans['rotate']),  np.cos(trans['rotate'])]])
            origin = (np.array(img.shape[:2])/2.0 - 0.5) + np.dot(matrix,origin - (np.array(inimg_shape)/2.0 - 0.5))
            
        # Finally, extract patch
        ptl = [int(np.round(i - self.patch_size/2)) for i in origin]
        pbr = [i + self.patch_size for i in ptl]
        img_patch = 128 + np.zeros([self.patch_size,self.patch_size,img.shape[2]],dtype=img.dtype)
        labels_patch = 255 + np.zeros([self.patch_size,self.patch_size],dtype=labels.dtype)
        real_ptl = [max(0,ind) for ind in ptl]
        real_pbr = [min(img.shape[i],ind) for i,ind in enumerate(pbr)]
        sub_ptl = [ -min(0,ind) for ind in ptl ]
        sub_pbr = [ self.patch_size - max(0,orig-real) for orig,real in zip(pbr,real_pbr) ]

        img_patch   [sub_ptl[0]:sub_pbr[0],sub_ptl[1]:sub_pbr[1],:] = img   [real_ptl[0]:real_pbr[0],real_ptl[1]:real_pbr[1],:]
        labels_patch[sub_ptl[0]:sub_pbr[0],sub_ptl[1]:sub_pbr[1]]   = labels[real_ptl[0]:real_pbr[0],real_ptl[1]:real_pbr[1]]

        return img_patch, labels_patch

def get_next_batch(name, batch_size):
	"""
		Checks
	"""
	max_ind = self.dataset.num_imgs()

	if name is None:
		name = "test"
	elif batch_size is None:
		batch_size = 50
