"""Training Sibling Network for heterogeneous face recognition
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
import argparse
import tensorflow as tf
import numpy as np

import utils
import tflib
from sibling_net import SiblingNetwork 

def main(args):
    # I/O
    config_file = args.config_file
    config = utils.import_file(config_file, 'config')

    trainset = utils.Dataset(config.train_dataset_path)
    testset = utils.Dataset(config.test_dataset_path)

    network = SiblingNetwork()
    network.initialize(config, trainset.num_classes)


    # Initalization for running
    log_dir = utils.create_log_dir(config, config_file)
    summary_writer = tf.summary.FileWriter(log_dir, network.graph)
    if config.restore_model is not None:
        network.restore_model(config.replace_scopes, config.restore_model, config.restore_scopes)

    # Separate type A and type B images for building batch and switch
    trainset.separate_AB()
    testset.separate_AB()

    # Set up test protocol and load images
    print('Loading images...')
    testset.images = utils.preprocess(testset.images, config, is_training=False)


    trainset.start_batch_queue(config, True)


    #
    # Main Loop
    #
    print('\nStart Training\nname: %s\n# epochs: %d\nepoch_size: %d\nbatch_size: %d\n'\
        % (config.name, config.num_epochs, config.epoch_size, config.batch_size))
    global_step = 0
    start_time = time.time()

    for epoch in range(config.num_epochs):

        if epoch == 0:
            print('Testing...')
            embeddings = network.extract_feature(testset.images, testset.is_typeB, config.batch_size)
            tars, fars, _ = utils.test_roc(embeddings, testset.labels, testset.is_typeB, FARs=[1e-5, 1e-4, 1e-3])
            with open(os.path.join(log_dir,'result.txt'),'at') as f:
                for i in range(len(tars)):
                    print('[%d] TAR: %2.4f FAR %2.5f' % (epoch+1, tars[i], fars[i]))
                    f.write('[%d] TAR: %2.4f FAR %2.5f\n' % (epoch+1, tars[i], fars[i]))
                    summary = tf.Summary()
                    summary.value.add(tag='test/tar_%d'%i, simple_value=tars[i])
                    summary_writer.add_summary(summary, global_step)

        # Training
        for step in range(config.epoch_size):
            # Prepare input
            learning_rate = utils.get_updated_learning_rate(global_step, config)
            batch = trainset.pop_batch_queue()
        
            wl, sm, global_step = network.train(batch['images'], batch['labels'], batch['is_typeB'], learning_rate, config.keep_prob)

            # Display
            if step % config.summary_interval == 0:
                duration = time.time() - start_time
                start_time = time.time()
                utils.display_info(epoch, step, duration, wl)
                summary_writer.add_summary(sm, global_step=global_step)

        # Testing
        print('Testing...')
        embeddings = network.extract_feature(testset.images, testset.is_typeB, config.batch_size)
        tars, fars, _ = utils.test_roc(embeddings, testset.labels, testset.is_typeB, FARs=[1e-5, 1e-4, 1e-3])
        with open(os.path.join(log_dir,'result.txt'),'at') as f:
            for i in range(len(tars)):
                print('[%d] TAR: %2.4f FAR %2.3f' % (epoch+1, tars[i], fars[i]))
                f.write('[%d] TAR: %2.4f FAR %2.3f\n' % (epoch+1, tars[i], fars[i]))
                summary = tf.Summary()
                summary.value.add(tag='test/tar_%d'%i, simple_value=tars[i])
                summary_writer.add_summary(summary, global_step)

        # Save the model
        network.save_model(log_dir, global_step)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="The path to the training configuration file",
                        type=str)
    args = parser.parse_args()
    main(args)
