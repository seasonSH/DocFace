"""Functions for building tensorflow graphs.
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

import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


def average_tensors(tensors, name=None):
    if len(tensors) == 1:
        return tf.identity(tensors[0], name=name)
    else:
        # Each tensor in the list should be of the same size
        expanded_tensors = []

        for t in tensors:
            expanded_t = tf.expand_dims(t, 0)
            expanded_tensors.append(expanded_t)

        average_tensor = tf.concat(axis=0, values=expanded_tensors)
        average_tensor = tf.reduce_mean(average_tensor, 0, name=name)

        return average_tensor

def average_grads(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of gradients. The outer list is over different 
        towers. The inner list is over the gradient calculation in each tower.
    Returns:
        List of gradients where the gradient has been averaged across all towers.
    """
    if len(tower_grads) == 1:
        return tower_grads[0]
    else:
        average_grads = []
        for grad_ in zip(*tower_grads):
            # Note that each grad looks like the following:
            #   (grad0_gpu0, ... , grad0_gpuN)
            average_grad = None if grad_[0]==None else average_tensors(grad_)
            average_grads.append(average_grad)

        return average_grads

def apply_gradient(update_gradient_vars, grads, optimizer, learning_rate, learning_rate_multipliers=None):
    assert(len(grads)==len(update_gradient_vars))
    if learning_rate_multipliers is None: learning_rate_multipliers = {}
    # Build a dictionary to save multiplier config
    # format -> {scope_name: ((grads, vars), lr_multi)}
    learning_rate_dict = {}
    learning_rate_dict['__default__'] = ([], 1.0)
    for scope, multiplier in learning_rate_multipliers.items():
        assert scope != '__default__'
        learning_rate_dict[scope] = ([], multiplier)

    # Scan all the variables, insert into dict
    scopes = learning_rate_dict.keys()
    for var, grad in zip(update_gradient_vars, grads):
        count = 0
        scope_temp = ''
        for scope in scopes:
            if scope in var.name:
                scope_temp = scope
                count += 1
        assert count <= 1, "More than one multiplier scopes appear in variable: %s" % var.name
        if count == 0: scope_temp = '__default__'
        if grad is not None:
            learning_rate_dict[scope_temp][0].append((grad, var))
     
    # Build a optimizer for each multiplier scope
    apply_gradient_ops = []
    print('\nLearning rate multipliers:')
    for scope, scope_content in learning_rate_dict.items():
        scope_grads_vars, multiplier = scope_content
        print('%s:\n  # variables: %d\n  lr_multi: %f' % (scope, len(scope_grads_vars), multiplier))
        if len(scope_grads_vars) == 0:
            continue
        scope_learning_rate = multiplier * learning_rate
        if optimizer=='ADAGRAD':
            opt = tf.train.AdagradOptimizer(scope_learning_rate)
        elif optimizer=='ADADELTA':
            opt = tf.train.AdadeltaOptimizer(scope_learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer=='ADAM':
            opt = tf.train.AdamOptimizer(scope_learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer=='RMSPROP':
            opt = tf.train.RMSPropOptimizer(scope_learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer=='MOM':
            opt = tf.train.MomentumOptimizer(scope_learning_rate, 0.9, use_nesterov=False)
        elif optimizer=='SGD':
            opt = tf.train.GradientDescentOptimizer(scope_learning_rate)
        else:
            raise ValueError('Invalid optimization algorithm')
        apply_gradient_ops.append(opt.apply_gradients(scope_grads_vars))
    print('')
    apply_gradient_op = tf.group(*apply_gradient_ops)

    return apply_gradient_op


def save_model(sess, saver, model_dir, global_step):
    with sess.graph.as_default():
        checkpoint_path = os.path.join(model_dir, 'ckpt')
        metagraph_path = os.path.join(model_dir, 'graph.meta')

        print('Saving variables...')
        saver.save(sess, checkpoint_path, global_step=global_step, write_meta_graph=False)
        if not os.path.exists(metagraph_path):
            print('Saving metagraph...')
            saver.export_meta_graph(metagraph_path)

def restore_model(sess, var_list, model_dir, restore_scopes=None, replace_rules=None):
    ''' Load the variable values from a checkpoint file into pre-defined graph.
    Filter the variables so that they contain at least one of the given keywords.'''
    with sess.graph.as_default():
        if restore_scopes is not None:
            var_list = [var for var in var_list if any([scope in var.name for scope in restore_scopes])]
        if replace_rules is not None:
            var_dict = {}
            for var in var_list:
                name_new = var.name
                for k,v in replace_rules.items(): name_new=name_new.replace(k,v)
                name_new = name_new[:-2] # When using dict, numbers should be removed
                var_dict[name_new] = var
            var_list = var_dict
        model_dir = os.path.expanduser(model_dir)
        ckpt_file = tf.train.latest_checkpoint(model_dir)

        print('Restoring {} variables from {} ...'.format(len(var_list), ckpt_file))
        saver = tf.train.Saver(var_list)
        saver.restore(sess, ckpt_file)

def load_model(sess, model_path, scope=None):
    ''' Load the the graph and variables values from a model path.
    Model path is either a a frozen graph or a directory with both
    a .meta file and checkpoint files.'''
    with sess.graph.as_default():
        model_path = os.path.expanduser(model_path)
        if (os.path.isfile(model_path)):
            # Frozen grpah
            print('Model filename: %s' % model_path)
            with gfile.FastGFile(model_path,'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
        else:
            # Load grapha and variables separatedly.
            meta_files = [file for file in os.listdir(model_path) if file.endswith('.meta')]
            assert len(meta_files) == 1
            meta_file = os.path.join(model_path, meta_files[0])
            ckpt_file = tf.train.latest_checkpoint(model_path)
            
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            saver = tf.train.import_meta_graph(meta_file, clear_devices=True, import_scope=scope)
            saver.restore(sess, ckpt_file)


def euclidean_distance(X, Y, sqrt=False):
    '''Compute the distance between each X and Y.

    Args: 
        X: a (m x d) tensor
        Y: a (d x n) tensor
    
    Returns: 
        diffs: an m x n distance matrix.
    '''
    with tf.name_scope('EuclideanDistance'):
        XX = tf.reduce_sum(tf.square(X), 1, keep_dims=True)
        YY = tf.reduce_sum(tf.square(Y), 0, keep_dims=True)
        XY = tf.matmul(X, Y)
        diffs = XX + YY - 2*XY
        diffs = tf.maximum(0.0, diffs)
        if sqrt == True:
            diffs = tf.sqrt(diffs)
    return diffs

def cosine_softmax(prelogits, label, num_classes, weight_decay, scale=16.0, reuse=None):
    ''' Tensorflow implementation of L2-Sofmax, proposed in:
        R. Ranjan, C. D. Castillo, and R. Chellappa. L2 constrained softmax loss for 
        discriminativeface veriﬁcation. arXiv:1703.09507, 2017. 
    '''
    
    nrof_features = prelogits.shape[1].value
    
    with tf.variable_scope('Logits', reuse=reuse):
        weights = tf.get_variable('weights', shape=(nrof_features, num_classes),
                regularizer=slim.l2_regularizer(weight_decay),
                initializer=slim.xavier_initializer(),
                # initializer=tf.truncated_normal_initializer(stddev=0.1),
                dtype=tf.float32)
        _scale = tf.get_variable('scale', shape=(),
                regularizer=slim.l2_regularizer(1e-2),
                initializer=tf.constant_initializer(1.00),
                trainable=True,
                dtype=tf.float32)

        weights_normed = tf.nn.l2_normalize(weights, dim=0)
        prelogits_normed = tf.nn.l2_normalize(prelogits, dim=1)

        if scale == 'auto':
            scale = tf.nn.softplus(_scale)
        else:
            assert type(scale) == float
            scale = tf.constant(scale)

        logits = scale * tf.matmul(prelogits_normed, weights_normed)


    cross_entropy =  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
            labels=label, logits=logits), name='cross_entropy')

    return logits, cross_entropy



def angular_softmax(prelogits, label, num_classes, global_step, weight_decay,
            m, lamb_min, lamb_max, reuse=None):
    ''' Tensorflow implementation of Angular-Sofmax, proposed in:
        W. Liu, Y. Wen, Z. Yu, M. Li, B. Raj, and L. Song. 
        Sphereface: Deep hypersphere embedding for face recognition. In CVPR, 2017.
    '''
    num_features = prelogits.shape[1].value
    batch_size = tf.shape(prelogits)[0]
    lamb_min = lamb_min
    lamb_max = lamb_max
    lambda_m_theta = [
        lambda x: x**0,
        lambda x: x**1,
        lambda x: 2.0*(x**2) - 1.0,
        lambda x: 4.0*(x**3) - 3.0*x,
        lambda x: 8.0*(x**4) - 8.0*(x**2) + 1.0,
        lambda x: 16.0*(x**5) - 20.0*(x**3) + 5.0*x
    ]

    with tf.variable_scope('AngularSoftmax', reuse=reuse):
        weights = tf.get_variable('weights', shape=(num_features, num_classes),
                regularizer=slim.l2_regularizer(1e-4),
                initializer=slim.xavier_initializer(),
                # initializer=tf.truncated_normal_initializer(stddev=0.1),
                trainable=True,
                dtype=tf.float32)
        lamb = tf.get_variable('lambda', shape=(),
                initializer=tf.constant_initializer(lamb_max),
                trainable=False,
                dtype=tf.float32)
        prelogits_norm  = tf.sqrt(tf.reduce_sum(tf.square(prelogits), axis=1, keep_dims=True))
        weights_normed = tf.nn.l2_normalize(weights, dim=0)
        prelogits_normed = tf.nn.l2_normalize(prelogits, dim=1)

        # Compute cosine and phi
        cos_theta = tf.matmul(prelogits_normed, weights_normed)
        cos_theta = tf.minimum(1.0, tf.maximum(-1.0, cos_theta))
        theta = tf.acos(cos_theta)
        cos_m_theta = lambda_m_theta[m](cos_theta)
        k = tf.floor(m*theta / 3.14159265)
        phi_theta = tf.pow(-1.0, k) * cos_m_theta - 2.0 * k

        cos_theta = cos_theta * prelogits_norm
        phi_theta = phi_theta * prelogits_norm

        lamb_new = tf.maximum(lamb_min, lamb_max/(1.0+0.1*tf.cast(global_step, tf.float32)))
        update_lamb = tf.assign(lamb, lamb_new)
        
        # Compute loss
        with tf.control_dependencies([update_lamb]):
            label_dense = tf.one_hot(label, num_classes, dtype=tf.float32)

            logits = cos_theta
            logits -= label_dense * cos_theta * 1.0 / (1.0+lamb)
            logits += label_dense * phi_theta * 1.0 / (1.0+lamb)
            
            cross_entropy =  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
                labels=label, logits=logits), name='cross_entropy')

    return cross_entropy


def am_softmax(prelogits, label, num_classes, 
                weight_decay, scale='auto', m=1.0, reuse=None):
    ''' Tensorflow implementation of AM-Sofmax, proposed in:
        F. Wang, W. Liu, H. Liu, and J. Cheng. Additive margin softmax for face veriﬁcation. arXiv:1801.05599, 2018.
    '''
    num_features = prelogits.shape[1].value
    batch_size = tf.shape(prelogits)[0]
    with tf.variable_scope('SplitSoftmax', reuse=reuse):
        weights = tf.get_variable('weights', shape=(num_classes, num_features),
                regularizer=slim.l2_regularizer(weight_decay),
                initializer=slim.xavier_initializer(),
                # initializer=tf.truncated_normal_initializer(stddev=0.0),
                # initializer=tf.constant_initializer(0),
                trainable=True,
                dtype=tf.float32)
        _scale = tf.get_variable('scale', shape=(),
                regularizer=slim.l2_regularizer(1e-2),
                initializer=tf.constant_initializer(1.00),
                trainable=True,
                dtype=tf.float32)

        # Normalizing the vecotors
        weights_normed = tf.nn.l2_normalize(weights, dim=1)
        prelogits_normed = tf.nn.l2_normalize(prelogits, dim=1)

        # Label and logits between batch and examplars
        label_mat_glob = tf.one_hot(label, num_classes, dtype=tf.float32)
        label_mask_pos_glob = tf.cast(label_mat_glob, tf.bool)
        label_mask_neg_glob = tf.logical_not(label_mask_pos_glob)

        dist_mat_glob = tf.matmul(prelogits_normed, tf.transpose(weights_normed))
        dist_pos_glob = tf.boolean_mask(dist_mat_glob, label_mask_pos_glob)
        dist_neg_glob = tf.boolean_mask(dist_mat_glob, label_mask_neg_glob)

        logits_glob = dist_mat_glob
        logits_pos_glob = tf.boolean_mask(logits_glob, label_mask_pos_glob)
        logits_neg_glob = tf.boolean_mask(logits_glob, label_mask_neg_glob)

        logits_pos = logits_pos_glob
        logits_neg = logits_neg_glob

        if scale == 'auto':
            # Automatic learned scale
            scale = tf.log(tf.exp(0.0) + tf.exp(_scale))
        else:
            # Assigned scale value
            assert type(scale) == float
            scale = tf.constant(scale)

        # Losses
        _logits_pos = tf.reshape(logits_pos, [batch_size, -1])
        _logits_neg = tf.reshape(logits_neg, [batch_size, -1])

        _logits_pos = _logits_pos * scale
        _logits_neg = _logits_neg * scale
        _logits_neg = tf.reduce_logsumexp(_logits_neg, axis=1)[:,None]

        loss_ = tf.nn.softplus(m + _logits_neg - _logits_pos)
        loss = tf.reduce_mean(loss_, name='am_softmax')


    return loss

def diam_softmax(prelogits, label, num_classes,
                    scale='auto', m=1.0, alpha=0.5, reuse=None):
    ''' Implementation of DIAM-Softmax, AM-Softmax with Dynamic Weight Imprinting (DWI), proposed in:
            Y. Shi and A. K. Jain. DocFace+: ID Document to Selfie Matching. arXiv:1809.05620, 2018.
        The weights in the DIAM-Softmax are dynamically updated using the mean features of training samples.
    '''
    num_features = prelogits.shape[1].value
    batch_size = tf.shape(prelogits)[0]
    with tf.variable_scope('AM-Softmax', reuse=reuse):
        weights = tf.get_variable('weights', shape=(num_classes, num_features),
                initializer=slim.xavier_initializer(),
                trainable=False,
                dtype=tf.float32)
        _scale = tf.get_variable('_scale', shape=(),
                regularizer=slim.l2_regularizer(1e-2),
                initializer=tf.constant_initializer(0.0),
                trainable=True,
                dtype=tf.float32)

        # Normalizing the vecotors
        prelogits_normed = tf.nn.l2_normalize(prelogits, dim=1)
        weights_normed = tf.nn.l2_normalize(weights, dim=1)

        # Label and logits between batch and examplars
        label_mat_glob = tf.one_hot(label, num_classes, dtype=tf.float32)
        label_mask_pos_glob = tf.cast(label_mat_glob, tf.bool)
        label_mask_neg_glob = tf.logical_not(label_mask_pos_glob)

        logits_glob = tf.matmul(prelogits_normed, tf.transpose(weights_normed))
        # logits_glob = -0.5 * euclidean_distance(prelogits_normed, tf.transpose(weights_normed))
        logits_pos_glob = tf.boolean_mask(logits_glob, label_mask_pos_glob)
        logits_neg_glob = tf.boolean_mask(logits_glob, label_mask_neg_glob)

        logits_pos = logits_pos_glob
        logits_neg = logits_neg_glob

        if scale == 'auto':
            # Automatic learned scale
            scale = tf.log(tf.exp(0.0) + tf.exp(_scale))
        else:
            # Assigned scale value
            assert type(scale) == float

        # Losses
        _logits_pos = tf.reshape(logits_pos, [batch_size, -1])
        _logits_neg = tf.reshape(logits_neg, [batch_size, -1])

        _logits_pos = _logits_pos * scale
        _logits_neg = _logits_neg * scale
        _logits_neg = tf.reduce_logsumexp(_logits_neg, axis=1)[:,None]

        loss_ = tf.nn.softplus(m + _logits_neg - _logits_pos)
        loss = tf.reduce_mean(loss_, name='diam_softmax')

        # Dynamic weight imprinting
        # We follow the CenterLoss to update the weights, which is equivalent to
        # imprinting the mean features
        weights_batch = tf.gather(weights, label)
        diff_weights = weights_batch - prelogits_normed
        unique_label, unique_idx, unique_count = tf.unique_with_counts(label)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])
        diff_weights = diff_weights / tf.cast(appear_times, tf.float32)
        diff_weights = alpha * diff_weights
        weights_update_op = tf.scatter_sub(weights, label, diff_weights)
        with tf.control_dependencies([weights_update_op]):
            weights_update_op = tf.assign(weights, tf.nn.l2_normalize(weights,dim=1))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, weights_update_op)
        
        return loss



def pair_loss(prelogits, label, num_classes, m=1.0, reuse=None):
    ''' Max-margin Pair Score (MPS) loss function proposed in:
        Y. Shi and A. K. Jain. DocFace: Matching ID Document Photos to Selfies. arXiv:1703.08388, 2017.
    '''
    nrof_features = prelogits.shape[1].value
    batch_size = tf.shape(prelogits)[0]
    with tf.variable_scope('MaxmarginPairLoss', reuse=reuse):

        # Normalizing the vecotors
        prelogits_normed = tf.nn.l2_normalize(prelogits, dim=1)

        prelogits_reshape = tf.reshape(prelogits_normed, [-1,2,tf.shape(prelogits_normed)[1]])
        prelogits_tmp = prelogits_reshape[:,0,:]
        prelogits_pro = prelogits_reshape[:,1,:]
    
        dist_mat_batch = -0.5 * euclidean_distance(prelogits_tmp, tf.transpose(prelogits_pro), True)
        # dist_mat_batch = tf.matmul(prelogits_tmp, tf.transpose(prelogits_pro))
        
        logits_mat_batch = dist_mat_batch

        num_pairs = tf.shape(prelogits_reshape)[0]
        label_mask_pos_batch = tf.cast(tf.eye(num_pairs), tf.bool)
        label_mask_neg_batch = tf.logical_not(label_mask_pos_batch)
        dist_pos_batch = tf.boolean_mask(dist_mat_batch, label_mask_pos_batch)
        dist_neg_batch = tf.boolean_mask(dist_mat_batch, label_mask_neg_batch)

        logits_pos_batch = tf.boolean_mask(logits_mat_batch, label_mask_pos_batch)
        logits_neg_batch = tf.boolean_mask(logits_mat_batch, label_mask_neg_batch)

        logits_pos = logits_pos_batch
        logits_neg = logits_neg_batch
    
        dist_pos = dist_pos_batch
        dist_neg = dist_neg_batch


        # Losses

        _logits_pos = tf.reshape(logits_pos, [num_pairs, -1])
        _logits_neg_1 = tf.reshape(logits_neg, [num_pairs, -1])
        _logits_neg_2 = tf.reshape(logits_neg, [-1, num_pairs])

        _logits_pos = _logits_pos
        _logits_neg_1 = tf.reduce_max(_logits_neg_1, axis=1)[:,None]
        _logits_neg_2 = tf.reduce_max(_logits_neg_2, axis=0)[:,None]
        _logits_neg = tf.maximum(_logits_neg_1, _logits_neg_2)
        # _logits_neg = tf.reduce_logsumexp(_logits_neg, axis=1)[:,None]

        loss_pos = tf.nn.relu(m + _logits_neg - _logits_pos) * 0.5
        loss_neg = tf.nn.relu(m + _logits_neg - _logits_pos) * 0.5
        loss = tf.reduce_mean(loss_pos + loss_neg, name='pair_loss')

    return loss



def pair_loss_sibling(prelogits_tmp, prelogits_pro, label_tmp, label_pro, 
                    num_classes, m=1.0, reuse=None):
    ''' Max-margin Pair Score (MPS) loss function proposed in:
        Y. Shi and A. K. Jain. DocFace: Matching ID Document Photos to Selfies. arXiv:1703.08388, 2017.
    '''
    num_features = prelogits_tmp.shape[1].value
    batch_size = tf.shape(prelogits_tmp)[0] + tf.shape(prelogits_pro)[0]
    with tf.variable_scope('MaxmarginPairLoss', reuse=reuse):

        # Normalizing the vecotors
        prelogits_tmp = tf.nn.l2_normalize(prelogits_tmp, dim=1)
        prelogits_pro = tf.nn.l2_normalize(prelogits_pro, dim=1)

        dist_mat_batch = -0.5 * euclidean_distance(prelogits_tmp, tf.transpose(prelogits_pro), True)   
        # dist_mat_batch = tf.matmul(prelogits_tmp, tf.transpose(prelogits_pro))
        
        logits_mat_batch = dist_mat_batch

        num_pairs = tf.shape(prelogits_tmp)[0]

        label_mask_pos_batch = tf.cast(tf.eye(num_pairs), tf.bool)
        label_mask_neg_batch = tf.logical_not(label_mask_pos_batch)
        dist_pos_batch = tf.boolean_mask(dist_mat_batch, label_mask_pos_batch)
        dist_neg_batch = tf.boolean_mask(dist_mat_batch, label_mask_neg_batch)

        logits_pos_batch = tf.boolean_mask(logits_mat_batch, label_mask_pos_batch)
        logits_neg_batch = tf.boolean_mask(logits_mat_batch, label_mask_neg_batch)

        logits_pos = logits_pos_batch
        logits_neg = logits_neg_batch

        # Losses
        _logits_pos = tf.reshape(logits_pos, [num_pairs, -1])
        _logits_neg_1 = tf.reshape(logits_neg, [num_pairs, -1])
        _logits_neg_2 = tf.reshape(logits_neg, [-1, num_pairs])

        _logits_neg_1 = tf.reduce_max(_logits_neg_1, axis=1)[:,None]
        _logits_neg_2 = tf.reduce_max(_logits_neg_2, axis=0)[:,None]
        _logits_neg = tf.maximum(_logits_neg_1, _logits_neg_2)

        loss_pos = tf.nn.relu(m + _logits_neg - _logits_pos) * 0.5
        loss_neg = tf.nn.relu(m + _logits_neg - _logits_pos) * 0.5
        loss = tf.reduce_mean(loss_pos + loss_neg, name='pair_loss')

    return loss
