# Import Modules
import os
import copy
import glob
import threading
import numpy as np
import scipy.io
import scipy.ndimage as spnd
import tensorflow as tf
from PIL import Image
import random as rn
# Pass in 'training.txt'
class AFLW_Dataset:
	# Constructors
    def __init__(self,data_partition,dataset):
        self.data_path = "/home/ekram67/Datasets"
       	self.version_number = ""
        self.dir = os.path.join(self.data_path,self.version_number)
        self.dataset = dataset
        self.name = '{}_{}'.format(self.dataset,self.version_number)
        self.all_idxs = open('{}/{}'.format(self.data_path,data_partition)).read().splitlines()
        # Return the total number of elements in the dataset
    def num_imgs(self):
        return len(self.all_idxs)
    # Convert PIL image to ndarray image
    def load_image(self, idx):
	if self.dataset == "lfw_5590":
		row_l = idx.split()
                im_name =  row_l[0]
		im_name = im_name[9:]
		im = Image.open(os.path.join(self.data_path,self.dataset,'{}'.format(im_name)))
                im = np.array(im, dtype=np.float32)
                if im.ndim < 3:
                        im_PIL = Image.fromarray(im)
                        im_PIL_RGB = im_PIL.convert('RGB')
                        im_PIL_RGB_ndarry = np.array(im_PIL_RGB,dtype=np.float32)
                	return im_PIL_RGB_ndarry
		return im
	else:
		row_l = idx.split()
        	im_name =  row_l[0]
		im = Image.open(os.path.join(self.data_path,'{}'.format(im_name)))
        	im = np.array(im, dtype=np.float32)
		if im.ndim < 3:
			im_PIL = Image.fromarray(im)
			im_PIL_RGB = im_PIL.convert('RGB')
			im_PIL_RGB_ndarry = np.array(im_PIL_RGB,dtype=np.float32)
			return im_PIL_RGB_ndarry
		return im
    def print_training_sample(idx):
        print('A sample from the training set is: {}'.format(idx))
    # Return the shape of the PIL image
    def image_shape(self,idx):
	row_l = idx.split()
        im_name =  row_l[0]
	im = Image.open(os.path.join(self.data_path,self.dataset,'{}'.format(im_name)))
        im = np.array(im, dtype=np.float32)
    # Extract landmarks
    def load_label_dict(self,idx):
        label_list = idx.split(" ")
	label_dict = {}
        label_dict['image'] = label_list[0]
        label_dict['image'] = label_dict['image'][5:]
        label_dict['features'] = label_list[2:]
        return label_dict
    # Unit variance, zero mean version of an image
    def preprocess_img(img):
        # use broadcasting to zero mean the matrix
        return (img - np.mean(img, axis=(0, 1)))/np.var(img,axix=(0,1))

class Data_Queue_tf:
    def __init__(self,dataset,landmark_num=10,gender_num = 2,smile_num = 2,glasses_num = 2,headpose_num = 5,imagesize=150,capacity=None,nthreads=1,batch_size=80,seed=5):
        if capacity is None:
            capcity = 10*batch_size
        self.capacity = capacity
        self.dataset = dataset
        self.nthreads = nthreads
	self.batch_size = batch_size
        self.imagesize = imagesize
	self.rnd = np.random.RandomState(seed)
	self.imagesize = imagesize
	self.landmark_num = landmark_num
	self.gender_num = gender_num
	self.glasses_num = glasses_num
	self.smile_num = smile_num
	self.headpose_num = headpose_num
	self.queue_input_img = tf.placeholder(tf.float32, shape=[self.imagesize,self.imagesize,3])
        self.queue_input_landmark = tf.placeholder(tf.float32, shape=[self.landmark_num])
        self.queue_input_gender = tf.placeholder(tf.float32, shape=[self.gender_num])
        self.queue_input_smile = tf.placeholder(tf.float32, shape=[self.smile_num])
        self.queue_input_glasses= tf.placeholder(tf.float32, shape=[self.glasses_num])
        self.queue_input_headpose = tf.placeholder(tf.float32, shape=[self.headpose_num])
#        self.rnd = np.random.RandomState(seed)
	self.queue = tf.FIFOQueue(capacity=capacity,
                                  dtypes=[np.float32, np.float32,np.float32,np.float32,np.float32,np.float32],
                                  # img and feature list
                                  shapes=[[self.imagesize,self.imagesize,3],
                                          [landmark_num],
					  [gender_num],
					  [smile_num],
					  [glasses_num],
					  [headpose_num]])
	self.enqueue_op = self.queue.enqueue([self.queue_input_img,
                                              self.queue_input_landmark,
                                              self.queue_input_gender,
					      self.queue_input_smile,
					      self.queue_input_glasses,
					      self.queue_input_headpose])
        self.dequeue_op = self.queue.dequeue()
        # Mutually exclusive lock
        self.cind_lock = threading.Lock()
    def get_input(self):
        batch = tf.train.batch(self.dequeue_op, batch_size=80,capacity=self.capacity, name='batch_'+self.dataset.name + '1')
        imgbatch_t,landmarkbatch_t,genderbatch_t,smilebatch_t,glassesbatch_t,headposebatch_t = batch
        return imgbatch_t,landmarkbatch_t,genderbatch_t,smilebatch_t,glassesbatch_t,headposebatch_t
    def start_threads(self,coord,sess,num_threads=5):
        self.cind = 0
        # create enqueuing thread
        enqueue_threads = [threading.Thread(target=self.enqueue,args=[coord,sess])for _ in range(num_threads)]
        for t in enqueue_threads:
            t.daemon = True
	    t.start()
        return enqueue_threads

    def enqueue(self,coord,sess):
	num_elements = 0
	while(True):
		max_ind = self.dataset.num_imgs()
        	# Lock acquired
        	with self.cind_lock:
                	if self.cind == 0:
                    		self.perm = self.rnd.permutation(self.dataset.num_imgs())
               	 	self.cind = (self.cind+1)%max_ind
			curr_ind = self.perm[self.cind]
			if curr_ind >2994:
				continue
        	# Load the image
        	cimg_idx = self.dataset.all_idxs[curr_ind]
        	curr = self.dataset.load_image(cimg_idx)
        	processed_img =  (curr - np.mean(curr, axis=(0, 1)))/np.var(curr,axis=(0,1))
        	# acquire features
        	curr_label_dict = self.dataset.load_label_dict(cimg_idx)
		i,j,k,l,m = self.feature_hot_encoding(curr_label_dict,cimg_idx)
		num_elements = num_elements + 1
		sess.run(self.enqueue_op,
                        feed_dict={self.queue_input_img:processed_img,
                        self.queue_input_landmark:i,
                        self.queue_input_gender:j,
                        self.queue_input_smile:k,
                        self.queue_input_glasses:l,
                        self.queue_input_headpose:m})
    def feature_hot_encoding(self,l_dict,cimg_idx):
	if len(l_dict['features']) == 14:
		landmark = np.array(l_dict['features'][0:10],dtype=np.float32)
        	if l_dict['features'][10] == 0:
            		gender = np.array([1.,0.],dtype=np.float32)
        	else:
            		gender = np.array([0.,1.],dtype=np.float32)
        	if l_dict['features'][11] == 0:
            		smile = np.array([1.,0.],dtype=np.float32)
        	else:
            		smile = np.array([0.,1.],dtype=np.float32)
        	if l_dict['features'][12] == 0:
            		glasses = np.array([1.,0.],dtype=np.float32)
        	else:
            		glasses = np.array([0.,1.],dtype=np.float32)
        	if l_dict['features'][13] == 0:
            		headpose = np.array([1.,0.,0.,0.,0.],dtype=np.float32)
        	elif l_dict['features'][13] == 1:
            		headpose = np.array([0.,1.,0.,0.,0.],dtype=np.float32)
        	elif l_dict['features'][13] == 2:
	    		headpose = np.arrat([0.,0.,1.,0.,0.],dtype=np.float32)
        	elif l_dict['features'][13] == 3:
            		headpose = np.array([0.,0.,0.,1.,0.],dtype=np.float32)
        	else:
            		headpose = np.array([0.,0.,0.,0.,1.],dtype=np.float32)
        	return landmark,gender,smile,glasses,headpose
	if len(l_dict['features']) == 13:
		print(cimg_idx)
		landmark = np.array(l_dict['features'][0:9],dtype=np.float32)
		landmark = np.insert(landmark,9,0)
		l_dict['features'] = np.insert(l_dict['features'],9,0)
                if l_dict['features'][10] == 0:
                        gender = np.array([1.,0.],dtype=np.float32)
                else:
                        gender = np.array([0.,1.],dtype=np.float32)
                if l_dict['features'][11] == 0:
                        smile = np.array([1.,0.],dtype=np.float32)
               	else:
                        smile = np.array([0.,1.],dtype=np.float32)
                if l_dict['features'][12] == 0:
                        glasses = np.array([1.,0.],dtype=np.float32)
               	else:
                        glasses = np.array([0.,1.],dtype=np.float32)
                if l_dict['features'][13] == 0:
                        headpose = np.array([1.,0.,0.,0.,0.],dtype=np.float32)
                elif l_dict['features'][13] == 1:
                        headpose = np.array([0.,1.,0.,0.,0.],dtype=np.float32)
                elif l_dict['features'][13] == 2:
                        headpose = np.arrat([0.,0.,1.,0.,0.],dtype=np.float32)
                elif l_dict['features'][13] == 3:
                        headpose = np.array([0.,0.,0.,1.,0.],dtype=np.float32)
               	else:
                        headpose = np.array([0.,0.,0.,0.,1.],dtype=np.float32)
                return landmark,gender,smile,glasses,headpose

