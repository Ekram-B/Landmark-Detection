import os
import copy
import glob
import threading
import numpy as np
import scipy.io
import scipy.ndimage as spnd
import tensorflow as tf
from PIL import Image

pascal_voc_datapath = '/data/cv/PASCAL-VOC/'

def get_palette(data_path=None):
    if data_path is None:
        data_path = pascal_voc_datapath
    # for paletting
    reference_idx = '2008_000666'
    palette = Image.open(os.path.join(data_path,'VOC2011','SegmentationClass','{}.png'.format(reference_idx)))
    return palette.palette
        
class VOCDataset:
    def __init__(self, data_set, release='VOC2011', data_path=None):
        if data_path is None:
            data_path = pascal_voc_datapath
        self.dir = os.path.join(data_path,release)

        self.palette = get_palette(data_path)
        self.name = '{}_{}'.format(release.lower(),data_set)
        self.release = release
        self.data_set = data_set
        self.classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person',
                        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        # Computed on VOC2011 Train
        self.class_probs = {'background':0.7480745536839799,
                            'aeroplane':0.0065929843719233,
                            'bicycle':0.0033117566131627882,
                            'bird':0.01051925155455967,
                            'boat':0.006668510650696824,
                            'bottle':0.007227213878861517,
                            'bus':0.01684281248823212, 
                            'car':0.015600101459773358, 
                            'cat':0.02540434686112562, 
                            'chair':0.011089005174509132, 
                            'cow':0.0060742973034393095, 
                            'diningtable':0.012648902526331982, 
                            'dog':0.015061443527794406, 
                            'horse':0.008588267160066087, 
                            'motorbike':0.012380973467488296, 
                            'person':0.04405926901108687, 
                            'pottedplant':0.005234272857682445, 
                            'sheep':0.00840898743593929, 
                            'sofa':0.012681131881438962, 
                            'train':0.013971851995449688, 
                            'tvmonitor':0.009560066096458434 }
#         self.counts = np.zeros(len(self.classes)+1,dtype=np.uint64)
#         self.loaded = set()
        self.weights = np.zeros(256,dtype=np.float32)
        for i,name in enumerate(self.classes):
            self.weights[i] = 1.0/self.class_probs[name]
        self.weights = len(self.classes) * self.weights / self.weights.sum()

        # get list of all files
        self.all_idxs = open('{}/ImageSets/Segmentation/{}.txt'.format(self.dir, data_set)).read().splitlines()

    def num_imgs(self):
        return len(self.all_idxs)

    def load_image(self, idx):
        im = Image.open(os.path.join(self.dir,'JPEGImages','{}.jpg'.format(idx)))
        im = np.array(im, dtype=np.float32)
        return im

    def load_label(self, idx):
        """
        Load label image
        """
        label = Image.open(os.path.join(self.dir,'SegmentationClass','{}.png'.format(idx)))
        label = np.array(label, dtype=np.uint8)

        return label

#    def palette(self, label_im):
#        '''
#        Transfer the VOC color palette to an output mask for visualization.
#        '''
#        if label_im.ndim == 3:
#            label_im = label_im[0]
#        label = Image.fromarray(label_im, mode='P')
#        label.palette = copy.copy(self.palette)
#        return label

class SBDataset:
    def __init__(self, data_set, data_path=None):
        self.name = 'sbd_{}'.format(data_set)
        self.data_set = data_set
        if data_path is None:
            data_path = pascal_voc_datapath
        self.palette = get_palette(data_path)
        self.dir = os.path.join(data_path,'SBD','dataset')
        self.classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person',
                        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        # Computed on SBD
        probs = [0.69576124, 0.00932639, 0.00903189, 0.00891391, 0.00691311, 
                 0.00556361, 0.01265280, 0.01986203, 0.03225846, 0.01289602, 
                 0.00628610, 0.00964067, 0.02888713, 0.00950409, 0.01187879, 
                 0.07519296, 0.00654954, 0.00622365, 0.01178860, 0.01330711, 0.00756189]
        self.class_probs = dict(zip(self.classes,probs))

        self.weights = np.zeros(256,dtype=np.float32)
        for i,name in enumerate(self.classes):
            self.weights[i] = 1.0/self.class_probs[name]
        self.weights = len(self.classes) * self.weights / self.weights.sum()

        # get list of all files
        self.all_idxs = open(os.path.join(self.dir,'{}.txt'.format(data_set))).read().splitlines()

    def num_imgs(self):
        return len(self.all_idxs)

    def load_image(self, idx):
        im = Image.open(os.path.join(self.dir,'img','{}.jpg'.format(idx)))
        im = np.array(im, dtype=np.float32)
        return im

    def load_label(self, idx):
        """
        Load label image
        """
        mat = scipy.io.loadmat(os.path.join(self.dir,'cls','{}.mat'.format(idx)))
        label = mat['GTcls'][0]['Segmentation'][0].astype(np.uint8)

        return label

class SemanticSegQueue:
    def __init__(self,dataset,augment=False,patch_size=96,batch_size=5,seed=0,capacity=None,num_batches=None):
        if capacity is None:
            capacity = 10*batch_size
        self.pad_size = patch_size
        self.dataset = dataset
        print('Dataset Queue: {} with {}'.format(dataset.name, dataset.num_imgs()))
        self.patch_size = patch_size
        # are used to feed data into our queue
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
            batch = tf.train.batch(self.dequeue_op, batch_size=batch_size,
                                   capacity=capacity,
                                   name='batch_'+dataset.name)
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

        # Mutually exclusive lock        
        self.cind_lock = threading.Lock()

    def _gen_transform(self,img):
        trans = {}
        # Augmenting operations
        if self.augment:
            medval = np.median(np.mean(img,axis=2))
            trans['flip'] = self.rnd.rand() > 0.5
            trans['rotate'] = np.pi*(2*self.rnd.rand()-1)
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
        return img_patch, labels_patch

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

#         p0 = self.rnd.randint(0,curr_img.shape[0]-self.patch_size+1)
#         p1 = self.rnd.randint(0,curr_img.shape[1]-self.patch_size+1)
#         curr_img = curr_img[p0:(p0+self.patch_size),p1:(p1+self.patch_size)]
#         curr_label = curr_label[p0:(p0+self.patch_size),p1:(p1+self.patch_size)]
        curr_weights = self.dataset.weights[curr_label]

        sess.run(self.enqueue_op, feed_dict={self.queue_input_data: curr_img.astype(np.float32),
                                             self.queue_input_target: curr_label.astype(np.int32),
                                             self.queue_input_weights: curr_weights.astype(np.float32)})        

    def enqueue(self,coord,sess):
#         with coord.stop_on_exception():
        while not coord.should_stop():
            self.load_next_batch(sess)

    def start_threads(self,coord,sess,num_threads=1):
        self.cind = 0
        enqueue_threads = [threading.Thread(target=self.enqueue, args=[coord,sess]) 
                           for _ in range(num_threads)]
        for t in enqueue_threads:
            t.daemon = True
            t.start()
        return enqueue_threads


