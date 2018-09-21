"""Basic Network for face recognition
"""
# MIT License
# 
# Copyright (c) 2017 Yichun Shi
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import time
import imp
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

import tflib

class BaseNetwork:
    def __init__(self):
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        tf_config = tf.ConfigProto(gpu_options=gpu_options,
                allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(graph=self.graph, config=tf_config)
            
    def initialize(self, config, num_classes):
        '''
            Initialize the graph from scratch according config.
        '''
        with self.graph.as_default():
            with self.sess.as_default():
                # Set up placeholders
                w, h = config.image_size
                channels = config.channels
                image_batch_placeholder = tf.placeholder(tf.float32, shape=[None, h, w, channels], name='image_batch')
                label_batch_placeholder = tf.placeholder(tf.int32, shape=[None], name='label_batch')
                learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
                keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob')
                phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
                global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')

                image_splits = tf.split(image_batch_placeholder, config.num_gpus)
                label_splits = tf.split(label_batch_placeholder, config.num_gpus)
                grads_splits = []
                split_dict = {}
                def insert_dict(k,v):
                    if k in split_dict: split_dict[k].append(v)
                    else: split_dict[k] = [v]
                        
                for i in range(config.num_gpus):
                    scope_name = '' if i==0 else 'gpu_%d' % i
                    with tf.name_scope(scope_name):
                        with tf.variable_scope('', reuse=i>0):
                            with tf.device('/gpu:%d' % i):
                                images = tf.identity(image_splits[i], name='inputs')
                                labels = tf.identity(label_splits[i], name='labels')
                                # Save the first channel for testing
                                if i == 0:
                                    self.inputs = images
                                
                                # Build networks
                                network = imp.load_source('network', config.network)
                                prelogits = network.inference(images, keep_prob_placeholder, phase_train_placeholder,
                                                        bottleneck_layer_size = config.embedding_size, 
                                                        weight_decay = config.weight_decay, 
                                                        model_version = config.model_version)
                                prelogits = tf.identity(prelogits, name='prelogits')
                                embeddings = tf.nn.l2_normalize(prelogits, dim=1, name='embeddings')
                                if i == 0:
                                    self.outputs = tf.identity(embeddings, name='outputs')

                                # Build all losses
                                losses = []

                                # Orignal Softmax
                                if 'softmax' in config.losses.keys():
                                    logits = slim.fully_connected(prelogits, num_classes, 
                                                                    weights_regularizer=slim.l2_regularizer(config.weight_decay),
                                                                    weights_initializer=slim.xavier_initializer(),
                                                                    biases_initializer=tf.constant_initializer(0.0),
                                                                    activation_fn=None, scope='Logits')
                                    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                    labels=labels, logits=logits), name='cross_entropy')
                                    losses.append(cross_entropy)
                                    insert_dict('sloss', cross_entropy)
                                # L2-Softmax
                                if 'cosine' in config.losses.keys():
                                    logits, cosine_loss = tflib.cosine_softmax(prelogits, labels, num_classes, 
                                                            weight_decay=config.weight_decay,
                                                            **config.losses['cosine']) 
                                    losses.append(cosine_loss)
                                    insert_dict('closs', cosine_loss)
                                # A-Softmax
                                if 'angular' in config.losses.keys():
                                    angular_loss = tflib.angular_softmax(prelogits, labels, num_classes, 
                                                            global_step, weight_decay=config.weight_decay,
                                                            **config.losses['angular'])  
                                    losses.append(angular_loss)
                                    insert_dict('aloss', angular_loss)
                                # AM-Softmax
                                if 'am' in config.losses.keys():
                                    am_loss = tflib.am_softmax(prelogits, labels, num_classes, 
                                                            weight_decay=config.weight_decay,
                                                            **config.losses['am'])
                                    losses.append(am_loss)
                                    insert_dict('loss', am_loss)
                                # Max-margin Pairwise Score (MPS)
                                if 'pair' in config.losses.keys():
                                    pair_loss = tflib.pair_loss(prelogits, labels, num_classes, 
                                                            **config.losses['pair'])  
                                    losses.append(pair_loss)
                                    insert_dict('loss', pair_loss)
                                # DIAM-Softmax
                                if 'diam' in config.losses.keys():
                                    diam_loss = tflib.diam_softmax(prelogits, labels, num_classes, 
                                                            **config.losses['diam'])  
                                    diam_loss = tf.identity(diam_loss, name='diam_loss')
                                    losses.append(diam_loss)
                                    insert_dict('amloss', diam_loss)

                               # Collect all losses
                                reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')
                                losses.append(reg_loss)
                                insert_dict('reg_loss', reg_loss)

                                total_loss = tf.add_n(losses, name='total_loss')
                                grads_split = tf.gradients(total_loss, tf.trainable_variables())
                                grads_splits.append(grads_split)



                # Merge the splits
                self.watchlist = {}
                grads = tflib.average_grads(grads_splits)
                for k,v in split_dict.items():
                    v = tflib.average_tensors(v)
                    self.watchlist[k] = v
                    if 'loss' in k:
                        tf.summary.scalar('losses/' + k, v)
                    else:
                        tf.summary.scalar(k, v)


                # Training Operaters
                apply_gradient_op = tflib.apply_gradient(tf.trainable_variables(), grads, config.optimizer,
                                        learning_rate_placeholder, config.learning_rate_multipliers)

                update_global_step_op = tf.assign_add(global_step, 1)

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                train_ops = [apply_gradient_op, update_global_step_op] + update_ops
                train_op = tf.group(*train_ops)

                tf.summary.scalar('learning_rate', learning_rate_placeholder)
                summary_op = tf.summary.merge_all()

                # Initialize variables
                self.sess.run(tf.local_variables_initializer())
                self.sess.run(tf.global_variables_initializer())
                self.saver = tf.train.Saver(tf.trainable_variables())

                # Keep useful tensors
                self.image_batch_placeholder = image_batch_placeholder
                self.label_batch_placeholder = label_batch_placeholder 
                self.learning_rate_placeholder = learning_rate_placeholder 
                self.keep_prob_placeholder = keep_prob_placeholder 
                self.phase_train_placeholder = phase_train_placeholder 
                self.global_step = global_step
                self.train_op = train_op
                self.summary_op = summary_op
                


    def train(self, image_batch, label_batch, learning_rate, keep_prob):
        feed_dict = {self.image_batch_placeholder: image_batch,
                    self.label_batch_placeholder: label_batch,
                    self.learning_rate_placeholder: learning_rate,
                    self.keep_prob_placeholder: keep_prob,
                    self.phase_train_placeholder: True,}
        _, wl, sm = self.sess.run([self.train_op, self.watchlist, self.summary_op], feed_dict = feed_dict)
        step = self.sess.run(self.global_step)

        return wl, sm, step
    
    def restore_model(self, *args, **kwargs):
        trainable_variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        tflib.restore_model(self.sess, trainable_variables, *args, **kwargs)

    def save_model(self, model_dir, global_step):
        tflib.save_model(self.sess, self.saver, model_dir, global_step)
        

    def load_model(self, *args, **kwargs):
        tflib.load_model(self.sess, *args, **kwargs)
        self.phase_train_placeholder = self.graph.get_tensor_by_name('phase_train:0')
        self.keep_prob_placeholder = self.graph.get_tensor_by_name('keep_prob:0')
        self.inputs = self.graph.get_tensor_by_name('inputs:0')
        self.outputs = self.graph.get_tensor_by_name('outputs:0')

    def extract_feature(self, images, batch_size, verbose=False):
        num_images = images.shape[0] if type(images)==np.ndarray else len(images)
        num_features = self.outputs.shape[1]
        result = np.ndarray((num_images, num_features), dtype=np.float32)
        start_time = time.time()
        for start_idx in range(0, num_images, batch_size):
            if verbose:
                elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))
                sys.stdout.write('# of images: %d Current image: %d Elapsed time: %s \t\r' 
                    % (num_images, start_idx, elapsed_time))
            end_idx = min(num_images, start_idx + batch_size)
            inputs = images[start_idx:end_idx]
            feed_dict = {self.inputs: inputs,
                        self.phase_train_placeholder: False,
                    self.keep_prob_placeholder: 1.0}
            result[start_idx:end_idx] = self.sess.run(self.outputs, feed_dict=feed_dict)
        if verbose:
            print('')
        return result

        
