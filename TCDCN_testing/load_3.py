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
	def __init__(self,data_partition):
        	self.data_path = "/home/ekram67/Datasets"
       		self.version_number = ""
        	self.dir = os.path.join(self.data_path,self.version_number)
        	self.dataset = "AFLW"
        	self.name = '{}_{}'.format(self.dataset,self.version_number)
        	self.all_idxs = open('{}/{}'.format(self.data_path,data_partition)).read().splitlines()
    # Return the total number of elements in the dataset
    	def num_imgs(self):
        	return len(self.all_idxs)
    # Convert PIL image to ndarray image
	def load_image(self, idx):
		row_l = idx.split()
                im_name =  row_l[0]
		im = Image.open(os.path.join(self.data_path,'{}'.format(im_name)))
        	im = np.array(im, dtype=np.float32)
		if im.ndim < 3:
			im_PIL = Image.fromarray(im)
			im_PIL_RGB = im_PIL.convert('RGB')
			im_PIL_RGB_ndarry = np.array(im_PIL_RGB,dtype=np.float32)
			print(im_PIL_RGB_ndarry.shape)
			return im_PIL_RGB_ndarry
		print(im.shape)
		return im
    # print training sample -- pass in all_idxs[x]
    	def print_training_sample(idx):
        	print('A sample from the training set is: {}'.format(idx))
    # Return the shape of the PIL image
    	def image_shape(self,idx):
		row_l = idx.split()
        	im_name =  row_l[0]
		im = Image.open(os.path.join(self.data_path,self.dataset,'{}'.format(im_name)))
        	im = np.array(im, dtype=np.float32)
        	print('The image idx is modeled by a ndarray of the shape {}'.format(im.shape))
    # Extract landmarks
    	def load_label(self,idx):
        	label_list = idx.split(" ")
		label_dict = {}
        	label_dict['image'] = label_list[0]
        	label_dict['image'] = label_dict['image'][5:]
        	label_dict['landmarks'] = label_list[2:]
        	return label_list, label_dict
class Data_Queue:
    def __init__(self,dataset,augment=False,patch_size=40,batch_size=5,seed=0,capacity=None,num_batches=None):
        if capacity is None:
            capcity = 10*batch_size
        self.pad_size = patch_size
        self.patch_size = patch_size
        self.dataset = dataset
        print('Dataset {} to queue with {} images'.format(dataset.name,dataset.num_imgs()))
        # Neural network is fully convolutional
        self.queue_input_data = tf.placeholder(tf.float32, shape=[self.patch_size,self.patch_size, 3])
        self.queue_input_target = tf.placeholder(tf.float32, shape=[self.patch_size, self.patch_size])

        self.queue = tf.FIFOQueue(capacity=capacity,
                                  dtypes=[tf.float32, tf.float32],
                                  shapes=[[self.patch_size,self.patch_size,3],
                                          [self.patch_size,self.patch_size]])

        self.enqueue_op = self.queue.enqueue([self.queue_input_data,
                                              self.queue_input_target])
        self.dequeue_op = self.queue.dequeue()
        self.rnd = np.random.RandomState(seed)
        self.augment = augment
        if num_batches is None:
            batch = tf.train.batch(self.dequeue_op, batch_size = batch_size,capacity = capacity, name='batch_'+dataset.name)
            self.data_batch, self.target_batch = batch
        else:
            self.data_batch, self.target_batch = [], []
            for i in range(num_batches):
                batch = tf.train.batch(self.dequeue_op, batch_size = batch_size,capacity = capacity, name='batch_'+dataset.name+'%02d'%i)
                self.data_batch.append(batch[0])
                self.target_batch.append(batch[1])
        self.cind_lock = threading.Lock()
    def enqueue(self,coord,dataset,sess):
        while not coord.should_stop():
            return self.load_next_batch(sess)
    def start_threads(self,coord,sess,num_threads=1):
        self.cind = 0
        enqueue_threads = [threading.Thread(target=self.enqueue,\
                             args=[coord,self.dataset,sess])for _ in range(num_threads)]
        for t in enqueue_threads:
            t.daemon = True
            t.start()
        return enqueue_threads
    def load_next_batch(self,sess):
        max_ind = self.dataset.num_imgs()
        # Lock acquired
        with self.cind_lock:
            if self.cind == 0:
                self.perm = self.rnd.permutation(self.dataset.num_imgs())
            curr_ind = self.perm[self.cind]
            self.cind = (self.cind+1)%max_ind
        cimg_idx = self.dataset.all_idxs[curr_ind]
        curr_img = self.dataset.load_image(cimg_idx)
        curr_label_list, curr_label_dict = self.dataset.load_label(cimg_idx)
	curr_label_list_new = [float(i) for i in curr_label_dict['landmarks']]
        img_patch, labels_patch = self._apply_transform(curr_img)
	print("Hello7")
        sess.run(self.enqueue_op, feed_dict={self.queue_input_data:img_patch,self.queue_input_target:labels_patch})
	print("Hello8")
	l_list, l_dict = self.dataset.load_label(cimg_idx)
        print("Hello9")
	landmark = np.array(l_dict['landmarks'][0:10],dtype=np.float32)
        if l_dict['landmarks'][10] == 1:
            gender = np.array([1.,0.],dtype=np.float32)
        else:
            gender = np.array([0.,1.],dtype=np.float32)
        if l_dict['landmarks'][11] == 1:
            smile = np.array([1.,0.],dtype=np.float32)
        else:
            smile = np.array([0.,1.],dtype=np.float32)
        if l_dict['landmarks'][12] == 1:
            glasses = np.array([1.,0.],dtype=np.float32)
        else:
            glasses = np.array([0.,1.],dtype=np.float32)
        if l_dict['landmarks'][13] == 1:
            headpose = np.array([1.,0.,0.,0.,0.],dtype=np.float32)
        elif l_dict['landmarks'][13] == 2:
            headpose = np.array([0.,1.,0.,0.,0.],dtype=np.float32)
        elif l_dict['landmarks'][13] == 3:
	    headpose = np.arrat([0.,0.,1.,0.,0.],dtype=np.float32)
        elif l_dict['landmarks'][13] == 4:
            headpose = np.array([0.,0.,0.,1.,0.],dtype=np.float32)
        else:
            headpose = np.array([0.,0.,0.,0.,1.],dtype=np.float32)
	print(landmark)
	print(gender)
	print(smile)
	print(glasses)
	print(headpose)
        return img_patch.reshape(-1,40,40),landmark,gender,smile,glasses,headpose

    def _gen_transform(self,img):
        trans = {}
        # Augmenting operations
        if self.augment:
            medval = np.median(np.mean(img,axis=2))
            trans['rotate'] = np.pi*(2*self.rnd.rand()-1)
            trans['contrast'] = self.rnd.lognormal(sigma=0.25)
            trans['brightness'] = np.clip(16*self.rnd.randn(),-medval,255-medval)
            trans['patch_origin'] = np.array([img.shape[i]/2.0 + rn.choice([-1,1])*(max(0,img.shape[i]/2 - self.patch_size/max(np.abs(np.cos(trans['rotate'])),np.abs(np.sin(trans['rotate']))))*(self.rnd.rand()-0.5)) for i in range(2)])
        else:
            trans['patch_origin'] = np.array([img.shape[i]/2.0 + rn.choice([-1,1])*(max(0,img.shape[i]/2 - self.patch_size)*(self.rnd.rand()-0.5)) for i in range(2)])
        return trans
    def _apply_transform(self,img,trans=None):
        if trans is None:
            trans = self._gen_transform(img)
        img = np.require(img,dtype=np.float32)
	print('{}'.format(type(img)))
#	print('{},{},{}'.format(img.shape[0],img.shape[1],img.shape[2]))
        # Colour augmentations
        if 'brightness' in trans:
            img = np.clip(img + trans['brightness'],0,255)
        if 'contrast' in trans:
            mean_rgb = np.mean(img,axis=(0,1),keepdims=True)
            img = np.clip(trans['contrast']*(img - mean_rgb) + mean_rgb,0,255)
        # Geometric augmentations
        origin = trans['patch_origin']
	if 'rotate' in trans:
            inimg_shape = img.shape[:2]
            img = spnd.interpolation.rotate(img,(180.0/np.pi)*trans['rotate'],mode='constant',cval=128,order=2,reshape=True)
            matrix = np.array([[ np.cos(trans['rotate']), -np.sin(trans['rotate'])],
                               [ np.sin(trans['rotate']),  np.cos(trans['rotate'])]])
            origin = (np.array(img.shape[:2])/2.0 - 0.5) + np.dot(matrix,origin - (np.array(inimg_shape)/2.0 - 0.5))

        # Finally, extract patch
        ptl = [int(np.round(i - self.patch_size/2)) for i in origin]
        pbr = [i + self.patch_size for i in ptl]
	print(self.patch_size)
        img_patch = 128 + np.zeros([self.patch_size,self.patch_size,img.shape[2]],dtype=img.dtype)
        labels_patch = 255 + np.zeros([self.patch_size,self.patch_size],dtype=np.float32)
        print("Hello")
	real_ptl = [max(0,ind) for ind in ptl]
	print("Hello2")
        real_pbr = [min(img.shape[i],ind) for i,ind in enumerate(pbr)]
        print("Hello3")
	sub_ptl = [ -min(0,ind) for ind in ptl ]
        print("Hello4")
	sub_pbr = [ self.patch_size - max(0,orig-real) for orig,real in zip(pbr,real_pbr) ]
        print("Hello5")
	img_patch   [sub_ptl[0]:sub_pbr[0],sub_ptl[1]:sub_pbr[1],:] = img   [real_ptl[0]:real_pbr[0],real_ptl[1]:real_pbr[1],:]
        print("Hello6")
	return img_patch, labels_patch
