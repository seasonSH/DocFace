"""Tensorflow implementation of Face-ResNet:
A. Hasnat, J. Bohne, J. Milgram, S. Gentric, and L. Chen. Deepvisage: Making face 
recognition simple yet with powerful generalization skills. arXiv:1703.08388, 2017.
"""
# MIT License
# 
# Copyright (c) 2018 Yichun Shi
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

def parametric_relu(x):
    num_channels = x.shape[-1].value
    with tf.variable_scope('p_re_lu'):
        alpha = tf.get_variable('alpha', (1,1,num_channels),
                        initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
        return tf.nn.relu(x) + alpha * tf.minimum(0.0, x)

# activation = lambda x: tf.keras.layers.PReLU(shared_axes=[1,2]).apply(x)
activation = parametric_relu

def se_module(input_net, ratio=16, reuse = None, scope = None):
    with tf.variable_scope(scope, 'SE', [input_net], reuse=reuse):
        h,w,c = tuple([dim.value for dim in input_net.shape[1:4]])
        assert c % ratio == 0
        hidden_units = int(c / ratio)
        squeeze = slim.avg_pool2d(input_net, [h,w], padding='VALID')
        excitation = slim.flatten(squeeze)
        excitation = slim.fully_connected(excitation, hidden_units, scope='se_fc1',
                                weights_regularizer=None,
                                # weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                weights_initializer=slim.xavier_initializer(), 
                                activation_fn=tf.nn.relu)
        excitation = slim.fully_connected(excitation, c, scope='se_fc2',
                                weights_regularizer=None,
                                # weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                weights_initializer=slim.xavier_initializer(), 
                                activation_fn=tf.nn.sigmoid)        
        excitation = tf.reshape(excitation, [-1,1,1,c])
        output_net = input_net * excitation

        return output_net
        

def conv_module(net, num_res_layers, num_kernels, reuse = None, scope = None):
    with tf.variable_scope(scope, 'conv', [net], reuse=reuse):
        # Every 2 conv layers constitute a residual block
        if scope == 'conv1':
            for i in range(len(num_kernels)):
                with tf.variable_scope('layer_%d'%i, reuse=reuse):
                    net = slim.conv2d(net, num_kernels[i], kernel_size=3, stride=1, padding='VALID',
                                    weights_initializer=slim.xavier_initializer())
                    print('| ---- layer_%d' % i)
            net = slim.max_pool2d(net, 2, stride=2, padding='VALID')
        else:
            shortcut = net
            for i in range(num_res_layers):
                with tf.variable_scope('layer_%d'%i, reuse=reuse):
                    net = slim.conv2d(net, num_kernels[0], kernel_size=3, stride=1, padding='SAME',
                                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    biases_initializer=None)
                    print('| ---- layer_%d' % i)
                if i % 2 == 1:
                    net = se_module(net)
                    net = net + shortcut
                    shortcut = net
                    print('| shortcut')
            # Pooling for conv2 - conv4
            if len(num_kernels) > 1:
                with tf.variable_scope('expand', reuse=reuse):
                    net = slim.conv2d(net, num_kernels[1], kernel_size=3, stride=1, padding='VALID',
                                    weights_initializer=slim.xavier_initializer())
                    net = slim.max_pool2d(net, 2, stride=2, padding='VALID')
                    print('- expand')

    return net

def inference(images, keep_probability, phase_train=True, bottleneck_layer_size=512, 
            weight_decay=0.0, reuse=None, model_version=None):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        activation_fn=activation,
                        normalizer_fn=None,
                        normalizer_params=None):
        with tf.variable_scope('FaceResNet', [images], reuse=reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=phase_train):
                print('input shape:', [dim.value for dim in images.shape])
                
                net = conv_module(images, 0, [32, 64], scope='conv1')
                print('module_1 shape:', [dim.value for dim in net.shape])

                net = conv_module(net, 2, [64, 128], scope='conv2')
                print('module_2 shape:', [dim.value for dim in net.shape])

                net = conv_module(net, 4, [128, 256], scope='conv3')
                print('module_3 shape:', [dim.value for dim in net.shape])

                net = conv_module(net, 10, [256, 512], scope='conv4')
                print('module_4 shape:', [dim.value for dim in net.shape])

                net = conv_module(net, 6, [512], scope='conv5')
                print('module_5 shape:', [dim.value for dim in net.shape])
                
                
                net = slim.flatten(net)
                net = slim.fully_connected(net, bottleneck_layer_size, scope='Bottleneck',
                                        weights_initializer=slim.xavier_initializer(), 
                                        activation_fn=None)

    return net
