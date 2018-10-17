"""Utilities for training and testing
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

import sys
import os
import numpy as np
from scipy import misc
import imp
import time
import math
import random
from datetime import datetime
import shutil
from multiprocessing import Process, Queue


# The images are separated into two types, A and B, for different branches in 
# the sibling networks. For example, in the ID-selfie problem, type A represents
# ID images, and type B stands for selfie images.
is_typeB = lambda x : os.path.basename(x).startswith('B')


def import_file(full_path_to_module, name='module.name'):
    
    module_obj = imp.load_source(name, full_path_to_module)
    
    return module_obj

def create_log_dir(config, config_file):
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(config.log_base_dir), config.name, subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    shutil.copyfile(config_file, os.path.join(log_dir,'config.py'))

    return log_dir

def create_sub_dir(log_dir, name):
    sub_dir = os.path.join(log_dir, name)
    if not os.path.isdir(sub_dir):
        os.makedirs(sub_dir)
    return sub_dir





class DataClass():
    def __init__(self, class_name, indices, label):
        self.class_name = class_name
        self.indices = list(indices)
        self.label = label
        self.indices_B = None
        self.indices_A = None
        return

    def random_pair(self):
        return np.random.permutation(self.indices_A)[:2]

    def random_AB_pair(self):
        assert self.indices_A is not None and self.indices_B is not None
        indices_A = np.random.permutation(self.indices_A)[0]
        indices_B = np.random.permutation(self.indices_B)[0]
        return [indices_A, indices_B]


class Dataset():

    def __init__(self, path=None):
        self.num_classes = None
        self.classes = None
        self.images = None
        self.labels = None
        self.features = None
        self.index_queue = None
        self.queue_idx = None
        self.batch_queue = None
        self.is_typeB =  None

        if path is not None:
            self.init_from_path(path)

    def init_from_path(self, path):
        path = os.path.expanduser(path)
        _, ext = os.path.splitext(path)
        if os.path.isdir(path):
            self.init_from_folder(path)
        elif ext == '.txt':
            self.init_from_list(path)
        else:
            raise ValueError('Cannot initialize dataset from path: %s\n\
                It should be either a folder or a .txt list file' % path)
        print('%d images of %d classes loaded' % (len(self.images), self.num_classes))

    def init_from_folder(self, folder):
        folder = os.path.expanduser(folder)
        class_names = os.listdir(folder)
        class_names.sort()
        classes = []
        images = []
        labels = []
        for label, class_name in enumerate(class_names):
            classdir = os.path.join(folder, class_name)
            if os.path.isdir(classdir):
                images_class = os.listdir(classdir)
                images_class.sort()
                images_class = [os.path.join(classdir,img) for img in images_class]
                indices_class = np.arange(len(images), len(images) + len(images_class))
                classes.append(DataClass(class_name, indices_class, label))
                images.extend(images_class)
                labels.extend(len(images_class) * [label])
        self.classes = np.array(classes, dtype=np.object)
        self.images = np.array(images, dtype=np.object)
        self.labels = np.array(labels, dtype=np.int32)
        self.num_classes = len(classes)

    def init_from_list(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        lines = [line.strip().split(' ') for line in lines]
        assert len(lines)>0, \
            'List file must be in format: "fullpath(str) label(int)"'
        images = [line[0] for line in lines]
        if len(lines[0]) > 1:
            labels = [int(line[1]) for line in lines]
        else:
            labels = [os.path.dirname(img) for img in images]
            _, labels = np.unique(labels, return_inverse=True)
        self.images = np.array(images, dtype=np.object)
        self.labels = np.array(labels, dtype=np.int32)
        self.init_classes()

    def init_classes(self):
        dict_classes = {}
        classes = []
        for i, label in enumerate(self.labels):
            if not label in dict_classes:
                dict_classes[label] = [i]
            else:
                dict_classes[label].append(i)
        for label, indices in dict_classes.items():
            classes.append(DataClass(str(label), indices, label))
        self.classes = np.array(classes, dtype=np.object)
        self.num_classes = len(classes)

    def separate_AB(self):
        assert type(self.images[0]) is str
        
        self.is_typeB = np.zeros(len(self.images), dtype=np.bool)

        for c in self.classes:
            # Find the index of type A file
            c.indices_A = [i for i in c.indices if not is_typeB(self.images[i])]
            assert len(c.indices_A) >= 1, str(self.images[c.indices])
            
            # Find the index of type B file
            c.indices_B = [i for i in c.indices if is_typeB(self.images[i])]
            assert len(c.indices_B) >= 1, str(self.images[c.indices])
            
            self.is_typeB[c.indices_B] = True

        print('type A images: %d   type B images: %d'%(np.sum(~self.is_typeB), np.sum(self.is_typeB)))


    # Data Loading
    def init_index_queue(self):
        if self.index_queue is None:
            self.index_queue = Queue()

        index_queue = np.random.permutation(len(self.images))[:,None]
        for idx in list(index_queue):
            self.index_queue.put(idx)


    def get_batch(self, batch_size, batch_format):
        ''' Get the indices from index queue and fetch the data with indices.'''
        indices_batch = []

        if batch_format == 'random_sample':
            while len(indices_batch) < batch_size:
                indices_batch.extend(self.index_queue.get(block=True, timeout=30))
            assert len(indices_batch) == batch_size
        elif batch_format == 'random_pair':
            assert batch_size%2 == 0
            classes = np.random.permutation(self.classes)[:batch_size//2]
            indices_batch = np.concatenate([c.random_pair() for c in classes], axis=0)
        elif batch_format == 'random_AB_pair':
            assert batch_size%2 == 0
            classes = np.random.permutation(self.classes)[:batch_size//2]
            indices_batch = np.concatenate([c.random_AB_pair() for c in classes], axis=0)
        else:
            raise ValueError('get_batch: Unknown batch_format: {}!'.format(batch_format))

        batch = {}
        if len(indices_batch) > 0:
            batch['images'] = self.images[indices_batch]
            batch['labels'] = self.labels[indices_batch]
            if self.is_typeB is not None:
                batch['is_typeB'] = self.is_typeB[indices_batch]
        return batch


    # Multithreading preprocessing images
    def start_index_queue(self):
        self.index_queue = Queue()
        def index_queue_worker():
            while True:
                if self.index_queue.empty():
                    self.init_index_queue()
                time.sleep(0.01)
        self.index_worker = Process(target=index_queue_worker)
        self.index_worker.daemon = True
        self.index_worker.start()

    def start_batch_queue(self, config, is_training, maxsize=1, num_threads=4):

        if self.index_queue is None:
            self.start_index_queue()

        self.batch_queue = Queue(maxsize=maxsize)
        def batch_queue_worker(seed):
            np.random.seed(seed)
            while True:
                batch = self.get_batch(config.batch_size, config.batch_format)
                if batch is not None:
                    batch['image_paths'] = batch['images']
                    batch['images'] = preprocess(batch['image_paths'], config, is_training)
                self.batch_queue.put(batch)

        self.batch_workers = []
        for i in range(num_threads):
            worker = Process(target=batch_queue_worker, args=(i,))
            worker.daemon = True
            worker.start()
            self.batch_workers.append(worker)


    def pop_batch_queue(self):
        batch = self.batch_queue.get(block=True, timeout=60)
        return batch

    def release_queue(self):
        if self.index_queue is not None:
            self.index_queue.close()
        if self.batch_queue is not None:
            self.batch_queue.close()
        if self.index_worker is not None:
            self.index_worker.terminate()
            del self.index_worker
            self.index_worker = None
        if self.batch_workers is not None:
            for w in self.batch_workers:
                w.terminate()
                del w
            self.batch_workers = None



# Calulate the shape for creating new array given (w,h)
def get_new_shape(images, size):
    w, h = tuple(size)
    shape = list(images.shape)
    shape[1] = h
    shape[2] = w
    shape = tuple(shape)
    return shape

def random_crop(images, size):
    n, _h, _w = images.shape[:3]
    w, h = tuple(size)
    shape_new = get_new_shape(images, size)
    assert (_h>=h and _w>=w)

    images_new = np.ndarray(shape_new, dtype=images.dtype)

    y = np.random.randint(low=0, high=_h-h+1, size=(n))
    x = np.random.randint(low=0, high=_w-w+1, size=(n))

    for i in range(n):
        images_new[i] = images[i, y[i]:y[i]+h, x[i]:x[i]+w]

    return images_new

def center_crop(images, size):
    n, _h, _w = images.shape[:3]
    w, h = tuple(size)
    assert (_h>=h and _w>=w)

    y = int(round(0.5 * (_h - h)))
    x = int(round(0.5 * (_w - w)))

    images_new = images[:, y:y+h, x:x+w]

    return images_new

def random_flip(images):
    images_new = images
    flips = np.random.rand(images_new.shape[0])>=0.5
    
    for i in range(images_new.shape[0]):
        if flips[i]:
            images_new[i] = np.fliplr(images[i])

    return images_new

def resize(images, size):
    n, _h, _w = images.shape[:3]
    w, h = tuple(size)
    shape_new = get_new_shape(images, size)

    images_new = np.ndarray(shape_new, dtype=images.dtype)

    for i in range(n):
        images_new[i] = misc.imresize(images[i], (h,w))

    return images_new

def standardize_images(images, standard):
    if standard=='mean_scale':
        mean = 127.5
        std = 128.0
    elif standard=='scale':
        mean = 0.0
        std = 255.0
    images_new = images.astype(np.float32)
    images_new = (images_new - mean) / std
    return images_new

def random_downsample(images, min_ratio):
    n, _h, _w = images.shape[:3]
    images_new = images
    ratios = min_ratio + (1-min_ratio) * np.random.rand(images_new.shape[0])

    for i in range(images_new.shape[0]):
        w = int(round(ratios[i] * _w))
        h = int(round(ratios[i] * _h))
        images_new[i,:h,:w] = misc.imresize(images[i], (h,w))
        images_new[i] = misc.imresize(images_new[i,:h,:w], (_h,_w))
        
    return images_new


def preprocess(images, config, is_training=False):
    # Load images first if they are file paths
    if type(images[0]) == str:
        image_paths = images
        images = []
        assert (config.channels==1 or config.channels==3)
        mode = 'RGB' if config.channels==3 else 'I'
        for image_path in image_paths:
            images.append(misc.imread(image_path, mode=mode))
        images = np.stack(images, axis=0)

    # Process images
    f = {
        'resize': resize,
        'random_crop': random_crop,
        'center_crop': center_crop,
        'random_flip': random_flip,
        'standardize': standardize_images,
        'random_downsample': random_downsample,
    }
    proc_funcs = config.preprocess_train if is_training else config.preprocess_test

    for proc in proc_funcs:
        proc_name, proc_args = proc[0], proc[1:]
        images = f[proc_name](images, *proc_args)
    if len(images.shape) == 3:
        images = images[:,:,:,None]
    return images
        


def get_updated_learning_rate(global_step, config):
    if config.learning_rate_strategy == 'step':
        max_step = -1
        learning_rate = 0.0
        for step, lr in config.learning_rate_schedule.items():
            if global_step >= step and step > max_step:
                learning_rate = lr
                max_step = step
        if max_step == -1:
            raise ValueError('cannot find learning rate for step %d' % global_step)
    elif config.learning_rate_strategy == 'cosine':
        initial = config.learning_rate_schedule['initial']
        interval = config.learning_rate_schedule['interval']
        end_step = config.learning_rate_schedule['end_step']
        step = math.floor(float(global_step) / interval) * interval
        assert step <= end_step
        learning_rate = initial * 0.5 * (math.cos(math.pi * step / end_step) + 1)
    return learning_rate

def display_info(epoch, step, duration, watch_list):
    sys.stdout.write('[%d][%d] time: %2.2f' % (epoch+1, step+1, duration))
    for item in watch_list.items():
        if type(item[1]) in [np.float32, np.float64]:
            sys.stdout.write('   %s: %2.3f' % (item[0], item[1]))
        elif type(item[1]) in [np.int32, np.int64, np.bool]:
            sys.stdout.write('   %s: %d' % (item[0], item[1]))
    sys.stdout.write('\n')



def get_pairwise_score_label(score_mat, label):
    n = label.size
    assert score_mat.shape[0]==score_mat.shape[1]==n
    triu_indices = np.triu_indices(n, 1)
    if len(label.shape)==1:
        label = label[:, None]
    label_mat = label==label.T
    score_vec = score_mat[triu_indices]
    label_vec = label_mat[triu_indices]
    return score_vec, label_vec

def test_roc(features, labels, is_typeB, FARs, return_index=False, score_mat=None):
    n = features.shape[0]
    assert labels.shape[0] == is_typeB.shape[0] == n
    if score_mat is None:
        score_mat = -euclidean(features, features)
    label_mat = labels.reshape([-1,1]) == labels.reshape([1,-1])
    is_pair = is_typeB.reshape([-1,1]) != is_typeB.reshape([1,-1])
    is_pair = np.triu(is_pair, 1)
    
    score_vec = score_mat[is_pair]
    label_vec = label_mat[is_pair]
    if return_index:
        rc = np.mgrid[0:n,0:n].transpose([1,2,0])
        rc_vec = rc[is_pair,:]
        TARs, FARs, thresholds, fai, fri = ROC(score_vec, label_vec,
            FARs=FARs, get_false_indices=True)
        for i in range(len(FARs)):
            fai[i], fri[i] = rc_vec[fai[i]], rc_vec[fri[i]]
        return TARs, FARs, thresholds, fai, fri
    else:
        TARs, FARs, thresholds = ROC(score_vec, label_vec,
            FARs=FARs, get_false_indices=False)
        return TARs, FARs, thresholds

def zero_one_switch(length):
    ''' Build a switch vector of the given length.'''
    assert length % 2 == 0
    zeros = np.zeros((length // 2, 1), dtype=np.bool)
    ones = np.ones((length // 2, 1), dtype=np.bool)
    switch = np.concatenate([zeros,ones], axis=1).reshape([-1])
    return switch

def euclidean(x1,x2):
    ''' Compute a distance matrix between every row of x1 and x2.'''
    assert x1.shape[1]==x2.shape[1]
    x2 = x2.transpose()
    x1_norm = np.sum(np.square(x1), axis=1, keepdims=True)
    x2_norm = np.sum(np.square(x2), axis=0, keepdims=True)
    dist = x1_norm + x2_norm - 2*np.dot(x1,x2)
    return dist

def normalize(x, ord=None, axis=None, epsilon=10e-12):
    ''' Devide the vectors in x by their norms.'''
    if axis is None:
        axis = len(x.shape) - 1
    norm = np.linalg.norm(x, ord=None, axis=axis, keepdims=True)
    x = x / (norm + epsilon)
    return x


def find_thresholds_by_FAR(score_vec, label_vec, FARs=None, epsilon=1e-8):
    assert len(score_vec.shape)==1
    assert score_vec.shape == label_vec.shape
    assert label_vec.dtype == np.bool
    score_neg = score_vec[~label_vec]
    score_neg[::-1].sort()
    # score_neg = np.sort(score_neg)[::-1] # score from high to low
    num_neg = len(score_neg)

    assert num_neg >= 1

    if FARs is None:
        thresholds = np.unique(score_neg)
        thresholds = np.insert(thresholds, 0, thresholds[0]+epsilon)
        thresholds = np.insert(thresholds, thresholds.size, thresholds[-1]-epsilon)
    else:
        FARs = np.array(FARs)
        num_false_alarms = np.round(num_neg * FARs).astype(np.int32)

        thresholds = []
        for num_false_alarm in num_false_alarms:
            if num_false_alarm==0:
                threshold = score_neg[0] + epsilon
            else:
                threshold = score_neg[num_false_alarm-1]
            thresholds.append(threshold)
        thresholds = np.array(thresholds)

    return thresholds

def ROC(score_vec, label_vec, thresholds=None, FARs=None, get_false_indices=False):
    ''' Compute Receiver operating characteristic (ROC) with a score and label vector.'''
    assert score_vec.ndim == 1
    assert score_vec.shape == label_vec.shape
    assert label_vec.dtype == np.bool
    
    if thresholds is None:
        thresholds = find_thresholds_by_FAR(score_vec, label_vec, FARs=FARs)

    assert len(thresholds.shape)==1 
    if np.size(thresholds) > 10000:
        print('number of thresholds (%d) very large, computation may take a long time!' % np.size(thresholds))

    # FARs would be check again
    TARs = np.zeros(thresholds.shape[0])
    FARs = np.zeros(thresholds.shape[0])
    false_accept_indices = []
    false_reject_indices = []
    for i,threshold in enumerate(thresholds):
        accept = score_vec >= threshold
        TARs[i] = np.mean(accept[label_vec])
        FARs[i] = np.mean(accept[~label_vec])
        if get_false_indices:
            false_accept_indices.append(np.argwhere(accept & (~label_vec)).flatten())
            false_reject_indices.append(np.argwhere((~accept) & label_vec).flatten())

    if get_false_indices:
        return TARs, FARs, thresholds, false_accept_indices, false_reject_indices
    else:
        return TARs, FARs, thresholds

def accuracy(score_vec, label_vec, thresholds=None):
    ''' Compute the accuracy given a binary label vector and a score vector.'''
    assert len(score_vec.shape)==1
    assert len(label_vec.shape)==1
    assert score_vec.shape == label_vec.shape
    assert label_vec.dtype==np.bool
    # find thresholds by TAR
    if thresholds is None:
        score_pos = score_vec[label_vec==True]
        thresholds = np.sort(score_pos)[::1]    

    assert len(thresholds.shape)==1
    if np.size(thresholds) > 10000:
        print('number of thresholds (%d) very large, computation may take a long time!' % np.size(thresholds))
    
    # Loop Computation
    accuracies = np.zeros(np.size(thresholds))
    for i, threshold in enumerate(thresholds):
        pred_vec = score_vec>=threshold
        accuracies[i] = np.mean(pred_vec==label_vec)

    argmax = np.argmax(accuracies)
    accuracy = accuracies[argmax]
    threshold = np.mean(thresholds[accuracies==accuracy])

    return accuracy, threshold
