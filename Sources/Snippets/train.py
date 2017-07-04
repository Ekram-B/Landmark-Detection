#!/usr/bin/env python3

import glob, sys, os, shutil, time
from datetime import datetime
import tensorflow as tf
import numpy as np
import threading
from queue import Queue

import pascal
from srcnn import SRFCN, FCN_VGG16, conf_mat_summaries

from importlib import reload
pascal = reload(pascal)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 6, 'Training batch size')
flags.DEFINE_integer('patch_size', 320, 'Training patch size')
flags.DEFINE_float('learning_rate',1e-4, 'Base learning rate')
flags.DEFINE_boolean('weighted_loss',True, 'Use class weighting to counteract imbalanced data')
# flags.DEFINE_string('device',None,'Which device use')
flags.DEFINE_integer('num_gpus',1,'How many GPUs to use')
flags.DEFINE_integer('num_rounds',1,'How many rounds per iteration to use')
flags.DEFINE_string('objective','fcn8','Objective function name')
flags.DEFINE_float('weight_decay',1e-10,'Amount of weight decay to use')
flags.DEFINE_float('keep_prob',0.5,'Dropout keep probability')
flags.DEFINE_boolean('from_scratch',False,'Start optimization from scratch')
flags.DEFINE_integer('save_interval',1000,'How frequently to checkpoint and validate')
flags.DEFINE_integer('print_interval',50,'How frequently to print to stdout')
flags.DEFINE_integer('summary_interval',50,'How frequently to dump extended summaries')
flags.DEFINE_string('exp_name','','Optional experiment name')
flags.DEFINE_string('voc_data',None,'Path to VOC data if needed')
flags.DEFINE_string('output_dir','/local/data/mab','Path to dump output data')
flags.DEFINE_string('network_type','fcn_vgg16','Type of network to use (fcn_vgg16 or srfcn)')

flags.DEFINE_boolean('vgg_init',False,'Use VGG16 to initialize (some) filters')
flags.DEFINE_string('act_fun','relu','Activation function to use in srfcn')
flags.DEFINE_integer('num_pyr_levels',4,'How many pyramid levels to use in srfcn')
flags.DEFINE_string('pyr_arch','pooled','Network architecture to use in the pyramid')
flags.DEFINE_string('gate_arch','sigmoid','Network architecture to use in the scale gating module')

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
        
        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
    
def io_worker(coord,q,writer):
    while not coord.should_stop():
        item = q.get()
        if item is None:
            break
        if isinstance(item,str) and item == 'flush':
            writer.flush()
        else:
            writer.add_summary(*item)
        q.task_done()

def get_run_name(FLAGS,network_name):
    batch_size = FLAGS.batch_size*FLAGS.num_gpus*FLAGS.num_rounds
    learning_rate = FLAGS.learning_rate
    weight_decay = FLAGS.weight_decay

    run_name = '{}_wd{}_eps{:.3}_batchsz{}'.format(network_name,
                                                   weight_decay,
                                                   learning_rate,
                                                   batch_size)
    if FLAGS.exp_name is not None and len(FLAGS.exp_name) > 0:
        run_name += '_{}'.format(FLAGS.exp_name)

    return run_name

def train():
    if FLAGS.voc_data is not None:
        pascal.pascal_voc_datapath = FLAGS.voc_data

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    #config.log_device_placement = True
    
    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()
    threads = []

    # setup validation data queue
    val_dataset = pascal.VOCDataset('seg11valid')
    num_val_imgs = val_dataset.num_imgs()
    val_batch_size = max([ i for i in range(1,5) if num_val_imgs % i == 0 ])
    val_queue = pascal.SemanticSegQueue(val_dataset,
                                        patch_size=500,
                                        batch_size=val_batch_size,
                                        capacity=1000,
                                        augment=False)
    extra_imgs = len(val_queue.dataset.all_idxs) % val_batch_size
    nvalbatches = int(len(val_queue.dataset.all_idxs) / val_batch_size) + (extra_imgs != 0)
    threads += val_queue.start_threads(coord,sess,num_threads=1)

    train_dataset = pascal.SBDataset('train')
    train_queue = pascal.SemanticSegQueue(train_dataset,
                                          patch_size=FLAGS.patch_size,
                                          batch_size=FLAGS.batch_size,
                                          capacity=500,
                                          augment=True,
                                          num_batches=FLAGS.num_gpus*FLAGS.num_rounds)
    num_classes = len(train_queue.dataset.classes)
    threads += train_queue.start_threads(coord,sess,num_threads=5)
    threads += tf.train.start_queue_runners(sess=sess, coord=coord)

    # Weight of a batch on the estimate after 1 epoch
    epoch_frac = 0.25
    num_its_per_epoch = float(train_dataset.num_imgs()) / (FLAGS.batch_size*FLAGS.num_gpus*FLAGS.num_rounds)
    exp_decay = np.exp(np.log(epoch_frac)/num_its_per_epoch)
    print('Targeting single epoch decay of {:.2f}, using per iteration decay of {:.4f}'.format(epoch_frac,exp_decay))

    with tf.device('/cpu:0'):
        iteration = tf.Variable(0,name='iteration',trainable=False)
        increment_iter = iteration.assign(iteration+1)
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        all_grads = []
        all_conf_mats = []
        all_avg_ces = []
        all_avg_wces = []
        all_objectives = []
#         all_train_nets = []
    
        with tf.variable_scope(FLAGS.network_type) as scope:
            if FLAGS.network_type == 'fcn_vgg16':
                val_net = FCN_VGG16()
                train_net = FCN_VGG16()
                kwargs = {'output_layers':FLAGS.objective,
                          'loss':FLAGS.objective,
                          'weighted':FLAGS.weighted_loss}
            elif FLAGS.network_type == 'srfcn':
                val_net = SRFCN()
                train_net = SRFCN()
                kwargs = {'num_levels':FLAGS.num_pyr_levels,
                          'act_fun':FLAGS.act_fun,
                          'vgg_init':FLAGS.vgg_init,
                          'structure':FLAGS.pyr_arch,
                          'gate_structure':FLAGS.gate_arch}
            for i in range(FLAGS.num_gpus*FLAGS.num_rounds):
                with tf.device('/gpu:%d'%(i%FLAGS.num_gpus)), tf.name_scope('training'):
#                     if FLAGS.network_type == 'fcn_vgg16':
#                         train_net = FCN_VGG16()
#                     elif FLAGS.network_type == 'srfcn':
#                         train_net = SRFCN()
#                     all_train_nets.append(train_net)
                    train_net.build(train_queue.data_batch[i],
                                    train_queue.target_batch[i],
                                    train_queue.weights_batch[i],
                                    num_classes=num_classes,
                                    train=True,
                                    **kwargs)
                    scope.reuse_variables()
                    if i == 0:
                        merged_imgs = tf.summary.merge_all(key='images')
                        merged_varsummaries = tf.summary.merge_all(key='variable_summaries')
                        merged_actsummaries = tf.summary.merge_all(key='activation_summaries')
                        merged_summaries = tf.summary.merge([a for a in [merged_imgs,merged_varsummaries,merged_actsummaries] if a is not None])
    
                    closs = train_net.loss

                    all_conf_mats.append(closs['conf_mat'])
        
                    weight_decay_reg = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
#                         tf.summary.scalar('weight_decay',weight_decay_reg)
        
                    all_avg_ces.append(closs['avg_ce'])
                    if FLAGS.weighted_loss:
                        all_avg_wces.append(closs['avg_wce'])
        
                    if FLAGS.weighted_loss:
                        objective = train_net.loss['avg_wce']
                    else:
                        objective = train_net.loss['avg_ce']
        
                    objective = objective + FLAGS.weight_decay*weight_decay_reg
                    all_objectives.append(objective)
                    
                    grads_and_vars = optimizer.compute_gradients(objective)
                    all_grads.append(grads_and_vars)
        
        with tf.device('/gpu:%d'%0), tf.name_scope('validation'):
            val_net.build(val_queue.data_batch,
                          val_queue.target_batch,
                          val_queue.weights_batch,
                          num_classes=num_classes,
                          train=False,
                          **kwargs)

            closs = val_net.loss

            valid_conf_mat = tf.placeholder(dtype=tf.float32,
                                            shape=closs['conf_mat'].shape)
            val_summaries = conf_mat_summaries(valid_conf_mat,True,['validation_summary'],
                                               class_names=train_queue.dataset.classes)

            tf.summary.scalar('avg_ce',closs['avg_ce'],collections=['validation_summary'])
            if FLAGS.weighted_loss:
                tf.summary.scalar('avg_wce',closs['avg_wce'],collections=['validation_summary'])

            merged_val = tf.summary.merge_all(key='validation_summary')

            tf.summary.scalar('avg_ce',closs['avg_ce'])

        with tf.name_scope('training'):
            tf.summary.scalar('avg_ce',tf.add_n(all_avg_ces)/len(all_avg_ces))
            if FLAGS.weighted_loss:
                tf.summary.scalar('avg_wce',tf.add_n(all_avg_wces)/len(all_avg_ces))
            tf.summary.scalar('objective',tf.add_n(all_objectives)/len(all_avg_ces))
            train_step = optimizer.apply_gradients(average_gradients(all_grads))
    
            smth_conf_mat = tf.Variable(tf.ones((num_classes,num_classes)),
                                        trainable=False,dtype=tf.float32)
            smth_conf_mat_upd_op = tf.assign(smth_conf_mat,
                                             exp_decay*smth_conf_mat + (1-exp_decay)*tf.add_n(all_conf_mats))
            summaries = conf_mat_summaries(smth_conf_mat_upd_op,True,
                                           class_names=train_queue.dataset.classes)
            

        print('Finished building network.')
    
        run_name = get_run_name(FLAGS,train_net.name)
        summary_dir = os.path.join(FLAGS.output_dir,'summaries','{}'.format(run_name))
        checkpoint_dir = os.path.join(FLAGS.output_dir,'checkpoints','{}'.format(run_name))
    
        if FLAGS.from_scratch:
            shutil.rmtree(summary_dir, ignore_errors=True)
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
        os.makedirs(summary_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        merged_sum = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(summary_dir,sess.graph)
        print('Finished setting up training, dumping output to: {}'.format(summary_dir))
    
        init = tf.global_variables_initializer()
    
    summary_io_q = Queue()
    t = threading.Thread(target=io_worker,args=[coord,summary_io_q,train_writer])
    t.daemon = True
    t.start()
    threads.append(t)
    
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1,max_to_keep=5)
    sess.run(init)
    if not FLAGS.from_scratch and os.path.exists(checkpoint_dir):
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint is not None:
            print ('Restoring parameters from previous checkpoint from {}'.format(latest_checkpoint))
            saver.restore(sess,latest_checkpoint)
        else:
            print ('No checkpoints found, starting from scratch')

    print('Training the network')
    total_it_time = 0
    num_timed_its = 0
#     train_feed_dict = dict([(train_net.dropout_keep_prob,FLAGS.keep_prob) for train_net in all_train_nets])
    train_feed_dict = {}
    while True:
        curr_it = sess.run(increment_iter)
        train_tic = time.time()
        do_extsummary = curr_it%FLAGS.summary_interval == 1
        if do_extsummary:
            _,cobjective,csummaries,csummary,cmerged_summaries = sess.run([train_step,objective,summaries,merged_sum,merged_summaries],
                                              feed_dict=train_feed_dict)
        else:
            _,cobjective,csummaries,csummary = sess.run([train_step,objective,summaries,merged_sum],
                                              feed_dict=train_feed_dict)
        it_time = time.time() - train_tic
        
        total_it_time += it_time
        num_timed_its += 1 
        summary_io_q.put((csummary,curr_it))
        if do_extsummary:
            summary_io_q.put((cmerged_summaries,curr_it))
            summary_io_q.put('flush')
        print("\r{:8d}: obj = {:8.2f}, ({:.2f}s/it)".format(curr_it,cobjective,total_it_time/num_timed_its),end='')
        print(', pixel_acc = {:.2f}, mean_acc = {:.2f}'.format(csummaries['pixel_acc'],csummaries['mean_acc']) + 20*' ',end='')
        if curr_it%FLAGS.print_interval == 0:
            print(20*' ')

        if FLAGS.save_interval>=0 and curr_it%FLAGS.save_interval == 0:
            total_conf_mat = None
            for i in range(nvalbatches):
                cconf_mat = sess.run(val_net.loss['conf_mat'])
                if i > 0:
                    total_conf_mat += cconf_mat
                else:
                    total_conf_mat = cconf_mat
     
                csummaries = sess.run(val_summaries,
                                      feed_dict={valid_conf_mat:total_conf_mat})
                print("\r  Running validation set: Batch {:05d} of {:05d}: ".format(i+1,nvalbatches),end='')
                for k in sorted(csummaries.keys()):
                    print('{} = {:.6f}  '.format(k,csummaries[k]),end='')
     
            cmerged_sum, csummaries = sess.run([merged_val,val_summaries],
                                               feed_dict={valid_conf_mat:total_conf_mat})
            summary_io_q.put((cmerged_sum,curr_it))
            summary_io_q.put('flush')
            print("\r    Validation Stats: ",end='')
            for k in sorted(csummaries.keys()):
                print('{} = {:.6f}  '.format(k,csummaries[k]),end='')
            print(50*' ')
    
            csave_path = saver.save(sess, checkpoint_dir+'/model.cpkt', global_step=curr_it)
            print("  Model saved in file: %s" % csave_path)


def main(argv=None):
    assert len(argv) <= 1
    train()


if __name__ == '__main__':
    tf.app.run()
