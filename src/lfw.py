"""Test protocols on LFW dataset
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

import os
import numpy as np
from collections import namedtuple

import utils

StandardFold = namedtuple('StandardFold', ['indices1', 'indices2', 'labels'])

class LFWTest:
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.images = None
        self.standard_folds = None
        self.queue_idx = None

    def init_standard_proto(self, lfw_pairs_file):
        index_dict = {}
        for i, image_path in enumerate(self.image_paths):
            image_name, image_ext = os.path.splitext(os.path.basename(image_path))
            index_dict[image_name] = i

        pairs = []
        with open(lfw_pairs_file, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)

        # 10 folds
        self.standard_folds = []
        for i in range(10):
            indices1 = np.zeros(600, dtype=np.int32)
            indices2 = np.zeros(600, dtype=np.int32)
            labels = np.array([True]*300+[False]*300, dtype=np.bool)
            # 300 positive pairs, 300 negative pairs in order
            for j in range(600):
                pair = pairs[600*i+j]
                if j < 300:
                    assert len(pair) == 3
                    img1 = pair[0] + '_' + '%04d' % int(pair[1])
                    img2 = pair[0] + '_' + '%04d' % int(pair[2])
                else:
                    assert len(pair) == 4
                    img1 = pair[0] + '_' + '%04d' % int(pair[1])
                    img2 = pair[2] + '_' + '%04d' % int(pair[3])                
                indices1[j] = index_dict[img1]
                indices2[j] = index_dict[img2]
            fold = StandardFold(indices1, indices2, labels)
            self.standard_folds.append(fold)

    def test_standard_proto(self, features):

        assert self.standard_folds is not None
        
        accuracies = np.zeros(10, dtype=np.float32)
        thresholds = np.zeros(10, dtype=np.float32)

        features1 = []
        features2 = []

        for i in range(10):
            # Training
            train_indices1 = np.concatenate([self.standard_folds[j].indices1 for j in range(10) if j!=i])
            train_indices2 = np.concatenate([self.standard_folds[j].indices2 for j in range(10) if j!=i])
            train_labels = np.concatenate([self.standard_folds[j].labels for j in range(10) if j!=i])

            train_features1 = features[train_indices1,:]
            train_features2 = features[train_indices2,:]
            
            train_score =  - np.sum(np.square(train_features1 - train_features2), axis=1)
            # train_score = np.sum(train_features1 * train_features2, axis=1)
            _, thresholds[i] = utils.accuracy(train_score, train_labels)

            # Testing
            fold = self.standard_folds[i]
            test_features1 = features[fold.indices1,:]
            test_features2 = features[fold.indices2,:]
            
            test_score = - np.sum(np.square(test_features1 - test_features2), axis=1)
            # test_score = np.sum(test_features1 * test_features2, axis=1)
            accuracies[i], _ = utils.accuracy(test_score, fold.labels, np.array([thresholds[i]]))

        accuracy = np.mean(accuracies)
        threshold = - np.mean(thresholds)
        return accuracy, threshold


