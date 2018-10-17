from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


model_params = {
    'conv1': ['conv1'],
    'conv12': ['conv1', 'conv2'],
    'conv13': ['conv1', 'conv2', 'conv3'],
    'conv14': ['conv1', 'conv2', 'conv3', 'conv4'],
    'conv15': ['conv1', 'conv2', 'conv3', 'conv4', 'conv5'],
    'all': ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc'],
    'conv45+fc': ['conv4', 'conv5', 'fc'],
    'conv5+fc': ['conv5', 'fc'],
    'fc': ['fc'],
}

batch_norm_params = {
    # Decay for the moving averages.
    'decay': 0.995,
    # epsilon to prevent 0s in variance.
    'epsilon': 0.001,
    # force in-place updates of mean and variance estimates
    'updates_collections': None,
    # Moving averages ends up in the trainable variables collection
    'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
}

batch_norm_params_last = {
    # Decay for the moving averages.
    'decay': 0.995,
    # epsilon to prevent 0s in variance.
    'epsilon': 10e-8,
    # force in-place updates of mean and variance estimates
    'center': False,
    # not use beta
    'scale': False,
    # not use gamma
    'updates_collections': None,
    # Moving averages ends up in the trainable variables collection
    'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
}

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
                    # net = activation(net)
                    print('| ---- layer_%d' % i)
            net = slim.max_pool2d(net, 2, stride=2, padding='VALID')
        else:
            shortcut = net
            for i in range(num_res_layers):
                with tf.variable_scope('layer_%d'%i, reuse=reuse):
                    net = slim.conv2d(net, num_kernels[0], kernel_size=3, stride=1, padding='SAME',
                                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    biases_initializer=None)
                    # net = activation(net)
                    print('| ---- layer_%d' % i)
                if i % 2 == 1:
                    net = se_module(net)
                    net = net + shortcut
                    shortcut = net
                    print('| shortcut')
            # Pooling for conv2 - conv4
            if len(num_kernels) > 1:
                with tf.variable_scope('expand', reuse=reuse):
                    # net = slim.batch_norm(net, **batch_norm_params)
                    net = slim.conv2d(net, num_kernels[1], kernel_size=3, stride=1, padding='VALID',
                                    weights_initializer=slim.xavier_initializer())
                    # net = activation(net)
                    net = slim.max_pool2d(net, 2, stride=2, padding='VALID')
                    print('- expand')

    return net

def build_scope(images, bottleneck_layer_size, shared_modules, scope_name, shared_scope_name, reuse=tf.AUTO_REUSE):
    get_scope = lambda x: shared_scope_name if x in shared_modules else scope_name
    with tf.variable_scope(get_scope('conv1'), reuse=reuse):
        print(tf.get_variable_scope().name)
        net = conv_module(images, 0, [32, 64], scope='conv1')
        print('module_1 shape:', [dim.value for dim in net.shape])
    with tf.variable_scope(get_scope('conv2'), reuse=reuse):
        print(tf.get_variable_scope().name)
        net = conv_module(net, 2, [64, 128], scope='conv2')
        print('module_2 shape:', [dim.value for dim in net.shape])
    with tf.variable_scope(get_scope('conv3'), reuse=reuse):
        print(tf.get_variable_scope().name)
        net = conv_module(net, 4, [128, 256], scope='conv3')
        print('module_3 shape:', [dim.value for dim in net.shape])
    with tf.variable_scope(get_scope('conv4'), reuse=reuse):
        print(tf.get_variable_scope().name)
        net = conv_module(net, 10, [256, 512], scope='conv4')
        print('module_4 shape:', [dim.value for dim in net.shape])
    with tf.variable_scope(get_scope('conv5'), reuse=reuse):
        print(tf.get_variable_scope().name)
        net = conv_module(net, 6, [512], scope='conv5')
        print('module_5 shape:', [dim.value for dim in net.shape])
    with tf.variable_scope(get_scope('fc'), reuse=reuse):
        print(tf.get_variable_scope().name)
        net = slim.flatten(net)
        prelogits = slim.fully_connected(net, bottleneck_layer_size, scope='Bottleneck',
                                weights_initializer=slim.xavier_initializer(), 
                                activation_fn=None)
    return prelogits

def inference(images_A, images_B, keep_probability=1.0, phase_train=True, bottleneck_layer_size=512, 
            weight_decay=0.0, reuse=None, model_version=None):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        activation_fn=activation,
                        normalizer_fn=None,
                        normalizer_params=None):
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=phase_train):
            with tf.variable_scope('FaceResNet', [images_A, images_B], reuse=reuse):
                
                shared_modules = model_params[model_version]

                print('input shape:', [dim.value for dim in images_A.shape])

                prelogits_A = build_scope(images_A, bottleneck_layer_size, 
                    shared_modules, "NetA", "SharedNet")
                prelogits_B = build_scope(images_B, bottleneck_layer_size, 
                    shared_modules, "NetB", "SharedNet")

        return prelogits_A, prelogits_B
