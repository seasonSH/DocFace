"""Extract features using pre-trained model
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
import sys
import time
import math
import argparse
import numpy as np
import tensorflow as tf

from basenet import BaseNetwork
from sibling_net import SiblingNetwork
import utils
import tflib


def main(args):
    # Get the configuration file
    config = utils.import_file(os.path.join(args.model_dir, 'config.py'), 'config')
    
    # Get the paths of the aligned images
    with open(args.image_list) as f:
        paths = [line.strip() for line in f]
    print('%d images to load.' % len(paths))
    assert(len(paths)>0)
    

    # Pre-process the images
    images = utils.preprocess(paths, config, False)
    switch = np.array([utils.is_typeB(p) for p in paths])
    print('%d type A images and %d type B images.' % (np.sum(switch), np.sum(~switch)))


    # Load model files and config file
    if config.use_sibling:
        network = SiblingNetwork()
    else:
        network = BaseNetwork()
    network.load_model(args.model_dir)


    # Run forward pass to calculate embeddings
    if config.use_sibling:
        embeddings = network.extract_feature(images, switch, args.batch_size, verbose=True)
    else:
        embeddings = network.extract_feature(images, args.batch_size, verbose=True)


    # Output the extracted features
    np.save(args.output, embeddings)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="The path to the pre-trained model directory",
                        type=str, required=True)
    parser.add_argument("--image_list", help="The list file to aligned face images to extract features",
                        type=str, required=True)
    parser.add_argument("--output", help="The output numpy file to store the extracted features",
                        type=str, required=True)
    parser.add_argument("--batch_size", help="Number of images per mini batch",
                        type=int, default=128)
    args = parser.parse_args()
    main(args)
