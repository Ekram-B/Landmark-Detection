from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from math import ceil
import sys

import numpy as np
import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68]
def voc_colormap(N=256):
    bitget = lambda val, idx: ((val & (1 << idx)) != 0)

    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= (bitget(c, 0) << 7 - j)
            g |= (bitget(c, 1) << 7 - j)
            b |= (bitget(c, 2) << 7 - j)
            c >>= 3

        cmap[i, :] = [r, g, b]
    return cmap

VOC_COLORMAP = voc_colormap()

def palette_impl(inpt):
    return np.squeeze(VOC_COLORMAP[inpt])

def put_kernels_on_grid (kernel, grid_Y, grid_X, pad = 1):
    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.

    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)

    Return:
      Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    '''

    x_max = tf.reduce_max(tf.abs(kernel))
    x_min = -x_max

    kernel1 = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x1 = tf.pad(kernel1, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad

    channels = kernel1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, channels])) #3

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, channels])) #3

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 255] and convert to uint8
    return tf.image.convert_image_dtype(x7, dtype = tf.uint8)

class VGG16(dict):
    def __init__(self,vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = sys.modules[self.__class__.__module__].__file__
            # print path
            path = os.path.abspath(os.path.join(path, os.pardir))
            # print path
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            logging.info("Load npy file from '%s'.", vgg16_npy_path)

        if not os.path.isfile(vgg16_npy_path):
            logging.error(("File '%s' not found. Download it from "
                           "ftp://mi.eng.cam.ac.uk/pub/mttt2/"
                           "models/vgg16.npy"), vgg16_npy_path)
            sys.exit(1)

        self.update(np.load(vgg16_npy_path, encoding='latin1').item())
vgg16_params = VGG16()


def equal_dims(shape1,shape2):
    if len(shape1) != len(shape2):
        return False
    for d1,d2 in zip(shape1,shape2):
        if isinstance(d1, tf.Dimension):
            d1 = d1.value
        if isinstance(d2, tf.Dimension):
            d2 = d2.value
        if d1 != d2:
            return False
    return True

def get_gauss_blur(ksz,sigma=np.sqrt(2)):
    x,y = np.meshgrid(np.arange(-ksz,ksz+1),np.arange(-ksz,ksz+1)) 
    gauss_kernel = np.exp(-(0.5/sigma**2)*(x**2 + y**2))
    gauss_kernel /= np.sum(gauss_kernel)
    return gauss_kernel.astype(np.float32)

def get_bilinear_filter(sz):
    f = np.ceil(np.asarray(sz)/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)

    bilinear = np.zeros(sz)
    for x in range(sz[0]):
        for y in range(sz[1]):
            value = (1 - abs(x / f[0] - c[0])) * (1 - abs(y / f[0] - c[0]))
            bilinear[x, y] = value

    return bilinear

def build_loss(num_classes,logits,labels,weights,num_levels=None):
    gauss_krn_nc = tf.constant(np.require(get_gauss_blur(1).reshape([3,3,1,1])*np.eye(num_classes).reshape([1,1,num_classes,num_classes]),
                     dtype=np.float32))
    gauss_krn_1c = tf.constant(np.require(get_gauss_blur(1).reshape([3,3,1,1]),dtype=np.float32))
    if num_levels is None:
        num_levels = len(logits)
    num_imgs = logits[0].shape[0].value
    patch_sz = logits[0].shape[1].value
    losses = []
    with tf.name_scope('loss') as scope:
        oh_labels = []
        if weights is not None:
            oh_weights = []

        for level in range(num_levels):
            curr_logits = logits[level]
            loss = {}
            losses.append(loss)

            if level == 0 and num_levels > 1:
                curr_ohlabels = tf.one_hot(labels,num_classes,dtype=tf.float32)
                if weights is not None:
                    curr_weights = tf.expand_dims(weights,-1)
            else:
                curr_ohlabels = tf.nn.conv2d(curr_ohlabels, gauss_krn_nc,
                                             [1, 2, 2, 1], padding='SAME')
                if weights is not None:
                    curr_weights = tf.nn.conv2d(curr_weights, gauss_krn_1c,
                                                [1, 2, 2, 1], padding='SAME')

            if level == 0:
                valid_targets = tf.less(labels,num_classes)
                # Set masked out labels to be zero to avoid NAN issues
                valid_labels = tf.where(valid_targets,labels,tf.zeros_like(labels))
                loss_vals = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=valid_labels,
                                                                           logits=curr_logits)
            else:
                valid_targets = tf.reduce_sum(curr_ohlabels,axis=3) >= 0.99
                loss_vals = tf.nn.softmax_cross_entropy_with_logits(labels=curr_ohlabels,
                                                                    logits=curr_logits)
            num_valid_targets = tf.count_nonzero(valid_targets,dtype=tf.float32,axis=[1,2])
            zero_loss = tf.zeros_like(loss_vals)
            valid_loss_vals = tf.where(valid_targets,loss_vals,zero_loss)
            
            if weights is not None:
                wce_per_img = tf.reshape(tf.matmul(tf.reshape(valid_loss_vals,[num_imgs, 1,-1]),
                                                   tf.reshape(   curr_weights,[num_imgs,-1, 1])),
                                         [num_imgs])
                wce = tf.reduce_mean(tf.divide(wce_per_img,num_valid_targets),name='wce_op')
                loss['avg_wce'] = wce
                
        
            ce_per_img = tf.reduce_sum(valid_loss_vals,axis=[1,2])
            avg_ce = tf.reduce_mean(tf.divide(ce_per_img,num_valid_targets),name='avg_ce_op')
            loss['avg_ce'] = avg_ce

            if level == 0:
                with tf.variable_scope('visualization'):
                    curr_preds = tf.cast(tf.argmax(curr_logits, dimension=3),dtype=tf.int32)
                    conf_mat = tf.confusion_matrix(tf.reshape(valid_labels,[-1]), 
                                                   tf.reshape(curr_preds,[-1]),
                                                   num_classes=num_classes,
                                                   weights=tf.reshape(valid_targets,[-1]),
                                                   dtype=tf.float32)
                    loss['conf_mat'] = conf_mat
            
                    preds_rgb = tf.py_func(palette_impl, [curr_preds], tf.uint8, stateful=False)
                    labels_rgb = tf.py_func(palette_impl, [labels], tf.uint8, stateful=False)
                    tf.summary.image('preds',preds_rgb,collections=['images'])
                    tf.summary.image('labels',labels_rgb,collections=['images'])
        

    return losses
    
def conf_mat_summaries(conf_mat,add_summaries=True,summary_collections=None,class_names=None):
    summaries = {}

    num_examples = tf.reduce_sum(conf_mat,name='num_examples_op')
    num_correct = tf.trace(conf_mat,name='num_correct_op')
    pixel_acc = tf.divide(num_correct,num_examples,
                          name='pixel_acc_op')
    summaries['pixel_acc'] = pixel_acc

    num_correct_by_class = tf.diag_part(conf_mat,name='num_correct_by_class_op')
    num_by_class = tf.reduce_sum(conf_mat,axis=1)
    class_acc = tf.divide(num_correct_by_class,
                          tf.add(num_by_class,1e-10),
                          name='class_acc_op')
    mean_acc = tf.reduce_mean(class_acc,
                              name='mean_acc_op')
    summaries['mean_acc'] = mean_acc
    
    union_by_class = num_by_class + tf.reduce_sum(conf_mat,axis=0) - num_correct_by_class
    iou_by_class = tf.divide(num_correct_by_class,
                            tf.add(union_by_class,1e-10),
                            name='iou_by_class_op')
    mean_iou = tf.reduce_mean(iou_by_class, name='mean_iou_op')
    summaries['mean_iou'] = mean_iou

    freqw_iou = tf.divide(tf.reduce_sum(tf.multiply(num_by_class,
                                                    iou_by_class)),
                          num_examples, name='freqw_iou_op')
    summaries['freqw_iou'] = freqw_iou
    
    if add_summaries:
        for name,val in summaries.items():
            tf.summary.scalar(name,val,collections=summary_collections)
        if class_names is not None:
            for i,name in enumerate(class_names):
                tf.summary.scalar('accuracy{:02d}_{}'.format(i,name),class_acc[i],collections=summary_collections)
    
    return summaries

class SRFCN:
    def __init__(self):
        self.debug = False
        self.variables = {}
    
    def _pyrbuild_vgg(self, train, act_fun, conv_init, with_maxpool=False):
        self._conv_layer('conv1_1',num_filts=64,act_fun=act_fun,init=conv_init,visualize_filters=True)
        self._conv_layer('conv1_2',num_filts=64,act_fun=act_fun,init=conv_init)
        if with_maxpool:
            self._max_pool('pool1',ksz=2)
        if train:
            self._dropout_layer('dropout1',self.dropout_keep_prob)

        self._conv_layer('conv2_1',num_filts=128,act_fun=act_fun,init=conv_init,
                         dilation=2)
        self._conv_layer('conv2_2',num_filts=128,act_fun=act_fun,init=conv_init,
                         dilation=2)
        if with_maxpool:
            self._max_pool('pool2',ksz=3)
        if train:
            self._dropout_layer('dropout2',self.dropout_keep_prob)

        self._conv_layer('conv3_1',
                         num_filts=256,
                         act_fun=act_fun,init=conv_init,
                         dilation=3)
        self._conv_layer('conv3_2',
                         num_filts=256,
                         act_fun=act_fun,init=conv_init,
                         dilation=3)
        self._conv_layer('conv3_3',
                         num_filts=256,
                         act_fun=act_fun,init=conv_init,
                         dilation=3)
        if with_maxpool:
            self._max_pool('pool3',ksz=4)

        self._conv_layer('conv4_1',
                         num_filts=512,
                         act_fun=act_fun,init=conv_init,
                         dilation=4)
        self._conv_layer('conv4_2',
                         num_filts=512,
                         act_fun=act_fun,init=conv_init,
                         dilation=4)
        self._conv_layer('conv4_3',
                         num_filts=512,
                         act_fun=act_fun,init=conv_init,
                         dilation=4)
        if with_maxpool:
            self._max_pool('pool3',ksz=5)

        self._conv_layer('conv5_1',
                         num_filts=512,
                         act_fun=act_fun,init=conv_init,
                         dilation=5)
        self._conv_layer('conv5_2',
                         num_filts=512,
                         act_fun=act_fun,init=conv_init,
                         dilation=5)
        self._conv_layer('conv5_3',
                         num_filts=512,
                         act_fun=act_fun,init=conv_init,
                         dilation=5)
        self.final_name = 'conv5_3'

    def _pyrbuild_skinny(self, train, act_fun, conv_init):
        self._conv_layer('conv1_1',num_filts=64,act_fun=act_fun,init=conv_init,visualize_filters=True)
        self._conv_layer('conv1_2',num_filts=64,act_fun=act_fun,init=conv_init)

        self._conv_layer('conv2_1',num_filts=64,act_fun=act_fun,init=conv_init)
        self._conv_layer('conv2_2',num_filts=64,act_fun=act_fun,init=conv_init)
        self._conv_layer('conv2_3',num_filts=64,act_fun=act_fun,init=conv_init)
        if train:
            self._dropout_layer('dropout2',self.dropout_keep_prob)

        self._conv_layer('conv3_1',
                         num_filts=64,
                         act_fun=act_fun,init=conv_init)
        self._conv_layer('conv3_2',
                         num_filts=64,
                         act_fun=act_fun,init=conv_init)
        self._conv_layer('conv3_3',
                         num_filts=64,
                         act_fun=act_fun,init=conv_init)

        if train:
            self._dropout_layer('dropout3',self.dropout_keep_prob)
        self._conv_layer('conv4_1',
                         num_filts=128,
                         act_fun=act_fun,init=conv_init)
        self._conv_layer('conv4_2',
                         num_filts=128,
                         act_fun=act_fun,init=conv_init)
        self.final_name = 'conv4_2'

    def _pyrbuild_pooled(self, train, act_fun, conv_init):
        self._conv_layer('conv1_1',num_filts=64,act_fun=act_fun,init=conv_init,visualize_filters=True)
        self._conv_layer('conv1_2',num_filts=64,act_fun=act_fun,init=conv_init)
        self._max_pool('pool1',ksz=2)

        self._conv_layer('conv2_1',num_filts=64,act_fun=act_fun,init=conv_init,dilation=2)
        self._conv_layer('conv2_2',num_filts=64,act_fun=act_fun,init=conv_init)
        self._conv_layer('conv2_3',num_filts=64,act_fun=act_fun,init=conv_init)
        self._max_pool('pool2',ksz=4)
        if train:
            self._dropout_layer('dropout2',self.dropout_keep_prob)

        self._conv_layer('conv3_1',
                         num_filts=64,
                         act_fun=act_fun,init=conv_init,dilation=4)
        self._conv_layer('conv3_2',
                         num_filts=64,
                         act_fun=act_fun,init=conv_init)
        self._conv_layer('conv3_3',
                         num_filts=64,
                         act_fun=act_fun,init=conv_init)

        if train:
            self._dropout_layer('dropout3',self.dropout_keep_prob)
        self._conv_layer('conv4_1',
                         num_filts=128,
                         act_fun=act_fun,init=conv_init)
        self._conv_layer('conv4_2',
                         num_filts=128,
                         act_fun=act_fun,init=conv_init)
        self.final_name = 'conv4_2'

    def _pyrbuild_dilated(self, train, act_fun, conv_init):
        self._conv_layer('conv1_1',num_filts=64,act_fun=act_fun,init=conv_init,visualize_filters=True)
        self._conv_layer('conv1_2',num_filts=64,act_fun=act_fun,init=conv_init)
#        self._conv_layer('conv1_3',num_filts=64,act_fun=act_fun,init=conv_init)
        self._max_pool('pool1',ksz=2)

        self._conv_layer('conv2_1',num_filts=64,act_fun=act_fun,init=conv_init,dilation=2)
        self._conv_layer('conv2_2',num_filts=64,act_fun=act_fun,init=conv_init,dilation=2)
        self._conv_layer('conv2_3',num_filts=64,act_fun=act_fun,init=conv_init,dilation=2)
        self._max_pool('pool2',ksz=4)
        if train:
            self._dropout_layer('dropout2',self.dropout_keep_prob)

        self._conv_layer('conv3_1',
                         num_filts=64,
                         act_fun=act_fun,init=conv_init,dilation=4)
        self._conv_layer('conv3_2',
                         num_filts=64,
                         act_fun=act_fun,init=conv_init,dilation=4)
        self._conv_layer('conv3_3',
                         num_filts=64,
                         act_fun=act_fun,init=conv_init,dilation=4)
        self._max_pool('pool3',ksz=8)
        if train:
            self._dropout_layer('dropout3',self.dropout_keep_prob)

        self._conv_layer('conv4_1',
                         num_filts=128,
                         act_fun=act_fun,init=conv_init,dilation=8)
        self._conv_layer('conv4_2',
                         num_filts=128,
                         act_fun=act_fun,init=conv_init,dilation=8)
        self.final_name = 'conv4_2'
                    
    def build(self,
              curr_rgb, curr_labels, curr_weights, num_classes, num_levels=3,
              weighted=True, train=False, act_fun='relu', vgg_init=False, 
              structure='pooled', gate_structure='sigmoid'):
        self.name = 'srfcn_pyr{}'.format(num_levels)
        if not weighted:
            self.name += '_unweighted'
        self.name += '_' + act_fun
        if vgg_init:
            self.name += '_vgginit'
        if structure is not None and len(structure) > 0:
            self.name += '_' + structure
        if gate_structure is not None and len(gate_structure) > 0:
            self.name += '_' + gate_structure

        self.num_classes = num_classes
        self.rgb = curr_rgb
        self.labels = curr_labels
        self.weights = curr_weights
        self.levels = []
        self.layers = {}
        self.stacked = []

        krn = np.require(get_gauss_blur(1).reshape([3,3,1,1])*np.eye(3).reshape([1,1,3,3]),
                         dtype=np.float32)
        gauss_krn = tf.constant(krn)
        
        if train:
#             self.dropout_keep_prob = tf.placeholder(tf.float32,shape=[],name="dropout_keep_prob")
            self.dropout_keep_prob = tf.placeholder_with_default(0.5,shape=[],name="dropout_keep_prob")

        if vgg_init:
            conv_init = 'vgg'
        else:
            conv_init = 'random'

        with tf.variable_scope('visualization'):
            tf.summary.image('input',self.rgb,collections=['images'])

        for level in range(num_levels):
            self._add_level('pyr{}'.format(level))

            with tf.name_scope('processing_'+self.levels[-1]):
                if level == 0:
                    mean_rgb = tf.reduce_mean(self.rgb,axis=(1,2,3),keep_dims=True)
                    nrm_rgb = (self.rgb - mean_rgb)
                    std_rgb = tf.sqrt(tf.maximum(tf.reduce_mean(tf.square(nrm_rgb)),tf.cast(tf.reduce_prod(tf.shape(nrm_rgb)[1:]),tf.float32)))
                    nrm_rgb = nrm_rgb/std_rgb
                else:
                    prev_rgb = nrm_rgb
                    nrm_rgb = tf.nn.conv2d(prev_rgb, gauss_krn,
                                           [1, 2, 2, 1], padding='SAME')
                self._append_layer(nrm_rgb,'nrm_rgb')

            with tf.name_scope(self.levels[-1]) as scope:
                if structure == 'vggmirror':
                    self._pyrbuild_vgg(train, act_fun, conv_init)
                elif structure == 'skinny':
                    self._pyrbuild_skinny(train, act_fun, conv_init)
                elif structure == 'pooled':
                    self._pyrbuild_pooled(train, act_fun, conv_init)
                elif structure == 'dilated':
                    self._pyrbuild_dilated(train, act_fun, conv_init)

        for level in range(num_levels-1,-1,-1):
            self._add_level('scale_recur_{}'.format(level))
            with tf.name_scope(self.levels[-1]) as scope:
                f_i   = self.layers[('pyr{}'.format(level),self.final_name)]
                if level == num_levels-1:
                    h_i = f_i
                else:
                    h_i_1 = self.layers[('scale_recur_{}'.format(level+1),'h')]
                    ht_i = self._upsample_layer('interp', bottom=h_i_1, out_shape=f_i.shape[1:3],
                                                uptype='bilinear')
                    if gate_structure == 'sigmoid':
                        g_i = self._conv_layer('gconv', bottom=[f_i,ht_i], filt_sz=1,
                                               num_filts=f_i.shape[3], act_fun='sigmoid',
                                               init='gating', visualize_activation=True)
                        h_i = g_i * f_i
                        h_i += (1-g_i) * ht_i
                    elif gate_structure == 'sigmoid1':
                        g_i = self._conv_layer('gconv', bottom=[f_i,ht_i], filt_sz=1,
                                               num_filts=1, act_fun='sigmoid',
                                               init='gating', visualize_activation=True)
                        h_i = g_i * f_i
                        h_i += (1-g_i) * ht_i
                    elif gate_structure == 'add':
                        h_i = ht_i + f_i
                    elif gate_structure == 'conv':
                        h_i = self._conv_layer('gconv', bottom=[f_i,ht_i], filt_sz=1,
                                               num_filts=f_i.shape[3], act_fun='linear',
                                               init='join',use_bias=False)
                self._append_layer(h_i,'h')
#                self._conv_layer('final_conv1',num_filts=128,init='random')
                self._conv_layer('logits',num_filts=num_classes,init='logits',
                                 act_fun='linear', visualize_activation=True)

        logits = [ self.layers[('scale_recur_{}'.format(i),'logits')] for i in range(num_levels) ]
        self.losses = build_loss(num_classes,logits,self.labels,self.weights if weighted else None)
        self.loss = {}
        self.loss['avg_ce'] = tf.add_n([closs['avg_ce']/(2**(2*i)) for i,closs in enumerate(self.losses)])
        if weighted:
            self.loss['avg_wce'] = tf.add_n([closs['avg_wce']/(2**(2*i)) for i,closs in enumerate(self.losses)])
        self.loss['conf_mat'] = self.losses[0]['conf_mat']

    def _add_level(self,name):
        self.levels.append(name)
        self.stacked.append([])

    def _append_layer(self,layer,name,level=-1):
        cname = (self.levels[level],name) 
        assert cname not in self.layers
        self.layers[cname] = layer

        self.stacked[level].append(cname)
        if self.debug:
            tf.Print(layer, [tf.shape(layer)],
                     message='Shape of layer {} ({}): '.format(len(self.stacked[-1]),cname),
                     summarize=4, first_n=1)

    def _get_bottom(self,level=-1):
        return self.layers[self.stacked[level][-1]]

    def _max_pool(self, name, stride=1, bottom=None, level=-1, ksz=2):
        if bottom is None:
            bottom = self._get_bottom()

        pool = tf.nn.max_pool(bottom, ksize=[1, ksz, ksz, 1],
                              strides=[1, stride, stride, 1],
                              padding='SAME', name=name)

        self._append_layer(pool, name, level)

        return pool

    def _conv_layer(self, name, num_filts=64, filt_sz=3, bottom=None,
                    level=-1, act_fun='elu', init='random',
                    dilation=None, visualize_filters=False,
                    visualize_activation=True, use_bias=True):
        if bottom is None:
            bottom = self._get_bottom()

        if init == 'vgg':
            filter_init = 'vgg'
            bias_init = 'vgg'
        elif init == 'random':
            filter_init = 'random'
            bias_init = 'zero'
        elif init == 'gating':
            filter_init = 'gating'
            bias_init = 'gating'
        elif init == 'join':
            filter_init = 'join'
            bias_init = 'zero'
        elif init == 'logits':
            filter_init = 'logits'
            bias_init = 'zero'
        
        if dilation is not None and dilation > 1:
            convop =  lambda bottom,filt: tf.nn.atrous_conv2d(bottom, filt, dilation, padding='SAME')
        else:
            convop =  lambda bottom,filt: tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
        with tf.variable_scope(name):
            if isinstance(bottom,(list,tuple)):
                for i,cbottom in enumerate(bottom):
                    bottom_shape = cbottom.shape
                    filt = self._get_conv_filter(name+'_{}'.format(i),bottom_shape[-1],num_filts,filt_sz,init=filter_init)
                    cconv = convop(cbottom, filt)
                    if i == 0:
                        filt_output = cconv
                    else:
                        filt_output += cconv
            else:
                bottom_shape = bottom.shape
                filt = self._get_conv_filter(name,bottom_shape[-1],num_filts,filt_sz,init=filter_init,visualize=visualize_filters)
                filt_output = convop(bottom, filt)

            if use_bias:
                conv_biases = self._get_bias(name,num_filts,init=bias_init)
                filt_output = tf.nn.bias_add(filt_output,conv_biases)

            if act_fun == 'relu':
                layer = tf.nn.relu(filt_output)
            elif act_fun == 'elu':
                layer = tf.nn.elu(filt_output)
            elif act_fun == 'sigmoid':
                layer = tf.sigmoid(filt_output)
            elif act_fun == 'linear':
                layer = filt_output
            else:
                assert 0, 'Unknown activation function'

        if visualize_activation:
            _activation_summary(layer,name)

        self._append_layer(layer, name, level)

        return layer

    def _dropout_layer(self, name, keep_prob, bottom=None, level=-1):
        if bottom is None:
            bottom = self._get_bottom()
        layer = tf.nn.dropout(bottom,keep_prob)
        self._append_layer(layer, name, level)

        return layer

    def _upsample_layer(self, name, bottom, out_shape=None, uptype='conv', num_filters=None, upsample_factor=2, filt_sz=2):
        in_shape = bottom.shape
        if num_filters is None:
            num_filters = in_shape[3] 
        if out_shape is None:
            out_shape = [ upsample_factor*in_shape[1].value,
                          upsample_factor*in_shape[2].value ]
        with tf.variable_scope(name):
            if uptype == 'conv':
                filters = self._get_interp_filter(name, filt_sz, in_shape[3], num_filters, init='identity')
                nn_interp = tf.image.resize_nearest_neighbor(bottom, out_shape)
                interp = tf.nn.atrous_conv2d(nn_interp, filters, upsample_factor, padding='SAME')
            elif uptype == 'nearest':
                interp = tf.image.resize_nearest_neighbor(bottom, out_shape)
            elif uptype == 'bilinear':
                interp = tf.image.resize_bilinear(bottom, out_shape)

        return interp

    def _get_variable(self,name,*args,return_status=False,**kwargs):
        idx = (tf.get_variable_scope().name,name)
        create_new = idx not in self.variables 
        if create_new:
            with tf.device('/cpu:0'):
                self.variables[idx] = tf.get_variable(name,*args,**kwargs)
        if return_status:
            return self.variables[idx], create_new
        else:
            return self.variables[idx]
        
    def _get_interp_filter(self, name, filt_sz, num_in, num_filters, init='bilinear'):
        f_shape = [filt_sz,filt_sz]
        if init == 'bilinear':
            weights = np.zeros(f_shape+[num_in,num_filters])
            bilinear = get_bilinear_filter(f_shape)
            for i in range(min([num_filters,num_in])):
                weights[:, :, i, i] = bilinear
        elif init == 'identity':
            assert num_filters == num_in
            weights = np.zeros(f_shape+[num_in,num_filters],dtype=np.float32)
            if filt_sz%2 == 0:
                for i in range(filt_sz):
                    for j in range(filt_sz):
                        weights[i,j,:,:] = np.eye(num_in,num_filters)/float(filt_sz*filt_sz)
            else:
                weights[(filt_sz-1)/2-1,(filt_sz-1)/2,:,:] = np.eye(num_in,num_filters)
#             for i in range(min([num_filters,num_in])):
#                 weights[:, :, i, i] = np.identity(filt_sz)
        else:
            assert False, 'Unknown initialization for interp_filter'

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)

        var,isnew = self._get_variable(name="interp", initializer=init,
                              shape=weights.shape,return_status=True)

        if isnew:
            weight_decay = tf.nn.l2_loss(var, name='weight_decay')
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 weight_decay)
            _variable_summaries(var,'ifilter')
        return var

    def _get_conv_filter(self, name, num_in, num_out, filt_sz, init, visualize=False):
        if isinstance(num_in,tf.Dimension):
            num_in = num_in.value
        shape = [filt_sz,filt_sz,num_in,num_out]
        if init == 'vgg':
            vgg_filt = vgg16_params.get(name,None)
            if vgg_filt is not None and equal_dims(vgg_filt[0].shape,shape):
                weights = vgg_filt[0]
                if name == 'conv1_1': # Flip conv1_1 filter to be RGB from the BGR format used by VGG16
                    weights = weights[:,:,[2,1,0],:]
                initializer = tf.constant_initializer(value=weights,
                                               dtype=tf.float32)
            else:
                print('Requested vgg_init but parameter missing or shapes dont match')
                print('  name = {}'.format(name))
                print('  required shape = {}'.format(shape))
                if vgg_filt is None:
                    print ('  parameter not found in dict')
                else:
                    print('  parameter shape = {}'.format(vgg_filt[0].shape))
#                 initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
                initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                             mode='FAN_AVG',
                                                                             uniform=False)
        elif init == 'random':
            if num_in == 3: # assume RGB in
                alpha = 0.95
                std_dev = np.sqrt(1.0/((num_in + num_out)/2.0))
                weights = std_dev*np.random.randn(*shape)
                avg_weights = std_dev*np.random.randn(filt_sz,filt_sz,1,num_out)
                init_weights = np.sqrt(alpha)*avg_weights + np.sqrt(1-alpha)*(weights - avg_weights)
                initializer = tf.constant_initializer(value=init_weights,
                                                      dtype=tf.float32)
#                 initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
            else:
#                 initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
                initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                             mode='FAN_AVG',
                                                                             uniform=False)
        elif init == 'gating':
#             initializer = tf.truncated_normal_initializer(stddev=0.01/(num_in*filt_sz**2))
            initializer = tf.contrib.layers.variance_scaling_initializer(factor=0.01,
                                                                         mode='FAN_IN',
                                                                         uniform=False)
        elif init == 'logits':
#             initializer = tf.truncated_normal_initializer(stddev=6.0/(num_in*filt_sz**2))
            initializer = tf.contrib.layers.variance_scaling_initializer(factor=0.01,
                                                                         mode='FAN_IN',
                                                                         uniform=False)
        elif init == 'join':
            weights = np.zeros(shape,dtype=np.float32)
            if filt_sz%2 == 0:
                for i in range(filt_sz):
                    for j in range(filt_sz):
                        weights[i,j,:,:] = np.eye(num_in,num_out)/float(filt_sz*filt_sz)
            else:
                weights[int((filt_sz-1)/2),int((filt_sz-1)/2),:,:] = np.eye(num_in,num_out)
            initializer = tf.constant_initializer(value=weights,
                                                  dtype=tf.float32)
        else:
            assert False, 'Unknown conv filter initialization procedure'

        var,isnew = self._get_variable(name="filter", initializer=initializer, shape=shape,
                                       return_status=True)
        
        if isnew:
            if visualize:
                with tf.variable_scope('visualization'):
                    grid_filters = put_kernels_on_grid(var,ceil(num_out/8),8,1)
                    tf.summary.image('filters',grid_filters,collections=['images','filters'])
                
            weight_decay = tf.nn.l2_loss(var, name='weight_decay')
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 weight_decay)

            _variable_summaries(var,'filter')
        return var

    def _get_bias(self, name, num, init):
        shape = [num]
        if init == 'vgg':
            vgg_filt = vgg16_params.get(name,None)
            if vgg_filt is not None and equal_dims(vgg_filt[1].shape,shape):
                initializer = tf.constant_initializer(value=vgg_filt[1],
                                               dtype=tf.float32)
            else:
                print('Requested vgg_init but parameter missing or shapes dont match')
                print('  name = {}'.format(name))
                print('  required shape = {}'.format(shape))
                if vgg_filt is None:
                    print ('  parameter not found in dict')
                else:
                    print('  paramter shape = {}'.format(vgg_filt[1].shape))
                initializer = tf.constant_initializer(0.0, dtype=tf.float32)
        elif init == 'zero':
            initializer = tf.constant_initializer(0.0, dtype=tf.float32)
        elif init == 'gating':
            initializer = tf.constant_initializer(0.0, dtype=tf.float32)
        else:
            assert False, 'Unknown bias initialization procedure'

        var,isnew = self._get_variable(name="bias", initializer=initializer, shape=shape,
                                 return_status=True)
        if isnew:
            _variable_summaries(var,'bias')
        return var

class FCN_VGG16:
    def __init__(self):
        self.data_dict = vgg16_params

    def build(self, curr_rgb, curr_labels, curr_weights, train=False, num_classes=20, random_init_fc8=True,
              debug=False, output_layers='all', loss='all', weighted=True):
        """
        Build the VGG model using loaded weights
        Parameters
        ----------
        rgb: image batch tensor
            Image in rgb shap. Scaled to Intervall [0, 255]
        train: bool
            Whether to build train or inference graph
        num_classes: int
            How many classes should be predicted (by fc8)
        random_init_fc8 : bool
            Whether to initialize fc8 layer randomly.
            Finetuning is required in this case.
        debug: bool
            Whether to print additional Debug Information.
        """
        assert output_layers in set(['fcn8','fcn16','fcn32','all'])
        assert loss in set(['fcn8','fcn16','fcn32','all'])
        
        self.name = 'fcn_vgg16'
        if weighted:
            self.name += '_w'
        self.name += loss

        self.num_classes = num_classes
        self.rgb = curr_rgb
        self.labels = curr_labels
        self.weights = curr_weights

        # Convert RGB to BGR
        with tf.name_scope('Processing'):

            red, green, blue = tf.split(self.rgb, 3, 3)
            # assert red.get_shape().as_list()[1:] == [224, 224, 1]
            # assert green.get_shape().as_list()[1:] == [224, 224, 1]
            # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
            bgr = tf.concat([
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ], 3)

            if debug:
                bgr = tf.Print(bgr, [tf.shape(bgr)],
                               message='Shape of input image: ',
                               summarize=4, first_n=1)

        self.conv1_1 = self._conv_layer(bgr, "conv1_1")
        self.conv1_2 = self._conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self._max_pool(self.conv1_2, 'pool1', debug)

        self.conv2_1 = self._conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self._conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self._max_pool(self.conv2_2, 'pool2', debug)

        self.conv3_1 = self._conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self._conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self._conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self._max_pool(self.conv3_3, 'pool3', debug)

        self.conv4_1 = self._conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self._conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self._conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self._max_pool(self.conv4_3, 'pool4', debug)

        self.conv5_1 = self._conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self._conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self._conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self._max_pool(self.conv5_3, 'pool5', debug)

        self.fc6 = self._fc_layer(self.pool5, "fc6")

        if train:
            self.dropout_keep_prob = tf.placeholder(tf.float32,shape=[],name="dropout_keep_prob")
            self.fc6 = tf.nn.dropout(self.fc6, self.dropout_keep_prob)

        self.fc7 = self._fc_layer(self.fc6, "fc7")
        if train:
            self.fc7 = tf.nn.dropout(self.fc7, self.dropout_keep_prob)

        if random_init_fc8:
            self.score_fr = self._score_layer(self.fc7, "score_fr",
                                              num_classes)
        else:
            self.score_fr = self._fc_layer(self.fc7, "score_fr",
                                           num_classes=num_classes,
                                           relu=False)

        if output_layers == 'all':
            out_fcn8, out_fcn16, out_fcn32 = True, True, True
        else:
            out_fcn8, out_fcn16, out_fcn32 = output_layers == 'fcn8', output_layers == 'fcn16', output_layers == 'fcn32'

        if out_fcn8 or out_fcn16:
            self.upscore2 = self._upscore_layer(self.score_fr,
                                                shape=tf.shape(self.pool4),
                                                num_classes=num_classes,
                                                debug=debug, name='upscore2',
                                                ksize=4, stride=2)
            self.score_pool4 = self._score_layer(self.pool4, "score_pool4",
                                                 num_classes=num_classes)
            self.fuse_pool4 = tf.add(self.upscore2, self.score_pool4)

        if out_fcn8:
            self.upscore4 = self._upscore_layer(self.fuse_pool4,
                                            shape=tf.shape(self.pool3),
                                            num_classes=num_classes,
                                            debug=debug, name='upscore4',
                                            ksize=4, stride=2)

            self.score_pool3 = self._score_layer(self.pool3, "score_pool3",
                                                 num_classes=num_classes)
            self.fuse_pool3 = tf.add(self.upscore4, self.score_pool3)
            self.upscore_fcn8 = self._upscore_layer(self.fuse_pool3,
                                             shape=tf.shape(bgr),
                                             num_classes=num_classes,
                                             debug=debug, name='upscore_fcn8',
                                             ksize=16, stride=8)
            self.uppred_fcn8 = tf.cast(tf.argmax(self.upscore_fcn8, dimension=3),tf.int32)

        if out_fcn16:
            self.upscore_fcn16 = self._upscore_layer(self.fuse_pool4,
                                                       shape=tf.shape(bgr),
                                                       num_classes=num_classes,
                                                       debug=debug, name='upscore_fcn16',
                                                       ksize=32, stride=16)
            self.uppred_fcn16 = tf.cast(tf.argmax(self.upscore_fcn16, dimension=3),tf.int32)

        if out_fcn32:
            self.upscore_fcn32 = self._upscore_layer(self.score_fr,
                                                       shape=tf.shape(bgr),
                                                       num_classes=num_classes,
                                                       debug=debug, name='upscore_fcn32',
                                                       ksize=64, stride=32)
            self.uppred_fcn32 = tf.cast(tf.argmax(self.upscore_fcn32, dimension=3),tf.int32)



        if any((out_fcn8, out_fcn16, out_fcn32)):
            self.predictions = {}
            self.scores = {}
            if out_fcn32:
                self.predictions['fcn32'] = self.uppred_fcn32
                self.scores['fcn32'] = self.upscore_fcn32
                self.prediction = self.predictions['fcn32']
                self.score = self.scores['fcn32']  
            if out_fcn16:
                self.predictions['fcn16'] = self.uppred_fcn16
                self.scores['fcn16'] = self.upscore_fcn16
                self.prediction = self.predictions['fcn16']
                self.score = self.scores['fcn16']  
            if out_fcn8:
                self.predictions['fcn8'] = self.uppred_fcn8
                self.scores['fcn8'] = self.upscore_fcn8
                self.prediction = self.predictions['fcn8']
                self.score = self.scores['fcn8']  



            losses = {}
            if loss == 'all':
                names = ['fcn8','fcn16','fcn32']
            else:
                names = [loss]
            for name in names:
                scores, preds = self.scores[name], self.predictions[name]
                losses[name] = build_loss(self.num_classes, scores, preds, self.labels,
                                          self.weights if weighted else None)

            self.losses = losses
            for k in ['fcn8','fcn16','fcn32']:
                if k in self.losses:
                    self.loss = self.losses[k]
                    break 

    def _max_pool(self, bottom, name, debug):
        pool = tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

        if debug:
            pool = tf.Print(pool, [tf.shape(pool)],
                            message='Shape of %s' % name,
                            summarize=4, first_n=1)
        return pool

    def _conv_layer(self, bottom, name):
        with tf.variable_scope(name) as scope:
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            # Add summary to Tensorboard
            _activation_summary(relu)
            return relu

    def _fc_layer(self, bottom, name, num_classes=None,
                  relu=True, debug=False):
        with tf.variable_scope(name) as scope:
            shape = bottom.get_shape().as_list()

            if name == 'fc6':
                filt = self.get_fc_weight_reshape(name, [7, 7, 512, 4096])
            elif name == 'score_fr':
                name = 'fc8'  # Name of score_fr layer in VGG Model
                filt = self.get_fc_weight_reshape(name, [1, 1, 4096, 1000],
                                                  num_classes=num_classes)
            else:
                filt = self.get_fc_weight_reshape(name, [1, 1, 4096, 4096])

            self._add_wd_and_summary(filt, "fc_wlosses")

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.get_bias(name, num_classes=num_classes)
            bias = tf.nn.bias_add(conv, conv_biases)

            if relu:
                bias = tf.nn.relu(bias)
            _activation_summary(bias)

            if debug:
                bias = tf.Print(bias, [tf.shape(bias)],
                                message='Shape of %s' % name,
                                summarize=4, first_n=1)
            return bias

    def _score_layer(self, bottom, name, num_classes):
        with tf.variable_scope(name) as scope:
            # get number of input channels
            in_features = bottom.get_shape()[3].value
            shape = [1, 1, in_features, num_classes]
            # He initialization Sheme
            if name == "score_fr":
                num_input = in_features
                stddev = (2 / num_input)**0.5
            elif name == "score_pool4":
                stddev = 0.001
            elif name == "score_pool3":
                stddev = 0.0001
            # Apply convolution
            w_decay = self.wd

            weights = self._variable_with_weight_decay(shape, stddev, w_decay,
                                                       decoder=True)
            conv = tf.nn.conv2d(bottom, weights, [1, 1, 1, 1], padding='SAME')
            # Apply bias
            conv_biases = self._bias_variable([num_classes], constant=0.0)
            bias = tf.nn.bias_add(conv, conv_biases)

            _activation_summary(bias)

            return bias

    def _upscore_layer(self, bottom, shape,
                       num_classes, name, debug,
                       ksize=4, stride=2):
        strides = [1, stride, stride, 1]
        with tf.variable_scope(name):
            in_features = bottom.get_shape()[3].value

            if shape is None:
                # Compute shape out of Bottom
                in_shape = tf.shape(bottom)

                h = ((in_shape[1] - 1) * stride) + 1
                w = ((in_shape[2] - 1) * stride) + 1
                new_shape = [in_shape[0], h, w, num_classes]
            else:
                new_shape = [shape[0], shape[1], shape[2], num_classes]
            output_shape = tf.stack(new_shape)

            logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
            f_shape = [ksize, ksize, num_classes, in_features]

            # create
            num_input = ksize * ksize * in_features / stride
            stddev = (2 / num_input)**0.5

            weights = self.get_deconv_filter(f_shape)
            self._add_wd_and_summary(weights, "fc_wlosses"+name)
            deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                            strides=strides, padding='SAME')

            if debug:
                deconv = tf.Print(deconv, [tf.shape(deconv)],
                                  message='Shape of %s' % name,
                                  summarize=4, first_n=1)

        _activation_summary(deconv)
        return deconv

    def get_deconv_filter(self, f_shape):
        width = f_shape[0]
        heigh = f_shape[0]
        f = ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        assert f_shape[2] <= f_shape[3], 'Size problem on deconv layer, filters would be useless (initialized to zero)'
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        var = tf.get_variable(name="up_filter", initializer=init,
                              shape=weights.shape)
        return var

    def get_conv_filter(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][0],
                                       dtype=tf.float32)
        shape = self.data_dict[name][0].shape
#        print('Layer name: %s' % name)
#        print('Layer shape: %s' % str(shape))
        var = tf.get_variable(name="filter", initializer=init, shape=shape)
        if not tf.get_variable_scope().reuse:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd,
                                       name='weight_loss')
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 weight_decay)
        _variable_summaries(var)
        return var

    def get_bias(self, name, num_classes=None):
        bias_wights = self.data_dict[name][1]
        shape = self.data_dict[name][1].shape
        if name == 'fc8':
            bias_wights = self._bias_reshape(bias_wights, shape[0],
                                             num_classes)
            shape = [num_classes]
        init = tf.constant_initializer(value=bias_wights,
                                       dtype=tf.float32)
        var = tf.get_variable(name="biases", initializer=init, shape=shape)
        _variable_summaries(var)
        return var

    def get_fc_weight(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][0],
                                       dtype=tf.float32)
        shape = self.data_dict[name][0].shape
        var = tf.get_variable(name="weights", initializer=init, shape=shape)
        if not tf.get_variable_scope().reuse:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd,
                                       name='weight_loss')
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 weight_decay)
        _variable_summaries(var)
        return var

    def _bias_reshape(self, bweight, num_orig, num_new):
        """ Build bias weights for filter produces with `_summary_reshape`

        """
        n_averaged_elements = num_orig//num_new
        avg_bweight = np.zeros(num_new)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx//n_averaged_elements
            if avg_idx == num_new:
                break
            avg_bweight[avg_idx] = np.mean(bweight[start_idx:end_idx])
        return avg_bweight

    def _summary_reshape(self, fweight, shape, num_new):
        """ Produce weights for a reduced fully-connected layer.

        FC8 of VGG produces 1000 classes. Most semantic segmentation
        task require much less classes. This reshapes the original weights
        to be used in a fully-convolutional layer which produces num_new
        classes. To archive this the average (mean) of n adjanced classes is
        taken.

        Consider reordering fweight, to perserve semantic meaning of the
        weights.

        Args:
          fweight: original weights
          shape: shape of the desired fully-convolutional layer
          num_new: number of new classes


        Returns:
          Filter weights for `num_new` classes.
        """
        num_orig = shape[3]
        shape[3] = num_new
        assert(num_new < num_orig)
        n_averaged_elements = num_orig//num_new
        avg_fweight = np.zeros(shape)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx//n_averaged_elements
            if avg_idx == num_new:
                break
            avg_fweight[:, :, :, avg_idx] = np.mean(
                fweight[:, :, :, start_idx:end_idx], axis=3)
        return avg_fweight

    def _variable_with_weight_decay(self, shape, stddev, wd, decoder=False):
        """Helper to create an initialized Variable with weight decay.

        Note that the Variable is initialized with a truncated normal
        distribution.
        A weight decay is added only if one is specified.

        Args:
          name: name of the variable
          shape: list of ints
          stddev: standard deviation of a truncated Gaussian
          wd: add L2Loss weight decay multiplied by this float. If None, weight
              decay is not added for this Variable.

        Returns:
          Variable Tensor
        """

        initializer = tf.truncated_normal_initializer(stddev=stddev)
        var = tf.get_variable('weights', shape=shape,
                              initializer=initializer)

        collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
        if wd is not None and (not tf.get_variable_scope().reuse):
            weight_decay = tf.multiply(
                tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection(collection_name, weight_decay)
        _variable_summaries(var)
        return var

    def _add_wd_and_summary(self, var, collection_name=None):
        if collection_name is None:
            collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
        if wd is not None and (not tf.get_variable_scope().reuse):
            weight_decay = tf.nn.l2_loss(var, name='weight_decay')
            tf.add_to_collection(collection_name, weight_decay)
        _variable_summaries(var)
        return var

    def _bias_variable(self, shape, constant=0.0):
        initializer = tf.constant_initializer(constant)
        var = tf.get_variable(name='biases', shape=shape,
                              initializer=initializer)
        _variable_summaries(var)
        return var

    def get_fc_weight_reshape(self, name, shape, num_classes=None):
#        print('Layer name: %s' % name)
#        print('Layer shape: %s' % shape)
        weights = self.data_dict[name][0]
        weights = weights.reshape(shape)
        if num_classes is not None:
            weights = self._summary_reshape(weights, shape,
                                            num_new=num_classes)
        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        var = tf.get_variable(name="weights", initializer=init, shape=shape)
        return var


def _activation_summary(var,name=None):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    if name is None:
        name = var.op.name
    with tf.name_scope('activation_summaries'):
        frac_nz = tf.nn.zero_fraction(var)
        mean = tf.reduce_mean(var)
        stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        minval = tf.reduce_min(var)
        maxval = tf.reduce_max(var)
    tf.summary.histogram(name + '/activations', var,
                         collections=['activation_summaries'])
    tf.summary.scalar(name + '/activations/sparsity', frac_nz,
                      collections=['activation_summaries'])
    tf.summary.scalar(name + '/activations/mean', mean,
                      collections=['activation_summaries'])
    tf.summary.scalar(name + '/activations/stddev', stddev,
                      collections=['activation_summaries'])
    tf.summary.scalar(name + '/activations/max', maxval,
                      collections=['activation_summaries'])
    tf.summary.scalar(name + '/activations/min', minval,
                      collections=['activation_summaries'])


def _variable_summaries(var,name=None):
    """Attach a lot of summaries to a Tensor."""
    if name is None:
        name = var.op.name
    mean = tf.reduce_mean(var)
    with tf.name_scope('variable_summaries'):
        stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.summary.scalar(name + '/mean', mean,
                      collections=['variable_summaries'])
    tf.summary.scalar(name + '/stddev', stddev,
                      collections=['variable_summaries'])
    tf.summary.scalar(name + '/max', tf.reduce_max(var),
                      collections=['variable_summaries'])
    tf.summary.scalar(name + '/min', tf.reduce_min(var),
                      collections=['variable_summaries'])
    tf.summary.histogram(name, var,
                         collections=['variable_summaries'])
