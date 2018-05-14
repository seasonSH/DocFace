''' Config Proto '''

import sys
import os


####### INPUT OUTPUT #######

# The name of the current model for output
name = 'faceres_ms_twin_test_1n'

# The folder to save log and model
log_base_dir = './log/'

# The interval between writing summary
summary_interval = 10

# Training dataset path
train_dataset_path = '/path/to/training/dataset/folder'

# Testing dataset path
train_dataset_path = '/path/to/testing/dataset/folder'

# LFW standard protocol file
lfw_pairs_file = './proto/lfw_pairs.txt'

# Target image size for the input of network
image_size = [96, 112]

# 3 channels means RGB, 1 channel for grayscale
channels = 3

# Preprocess for training
preprocess_train = [
    # ['resize', (48,56)],
    # ['resize', (96,112)],
    ['random_flip'],
    # ('random_crop', (96,112)],
    # ['random_downsample', 0.5],
    ['standardize', 'mean_scale'],
]

# Preprocess for testing
preprocess_test = [
    # ['resize', (96,112)],
    # ['center_crop', (96, 112)],
    ['standardize', 'mean_scale'],
]

# Number of GPUs
num_gpus = 1


####### NETWORK #######

# Use sibling network
use_sibling = True

# The network architecture
network = "nets/face_resnet.py"

# Model version, only for some networks
model_version = None

# Number of dimensions in the embedding space
embedding_size = 512


####### TRAINING STRATEGY #######

# Optimizer
optimizer = "MOM"

# Number of samples per batch
batch_size = 256

# Number of batches per epoch
epoch_size = 100

# Number of epochs
num_epochs = 8

# learning rate strategy
learning_rate_strategy = 'step'

# learning rate schedule
lr = 0.01
learning_rate_schedule = {
    0:      1 * lr,
    500:    0.1 * lr,
}

# Multiply the learning rate for variables that contain certain keywords
learning_rate_multipliers = {
}

# The model folder from which to retore the parameters
restore_model = '/path/to/the/pretrained/model/folder'

# Keywords to filter restore variables, set None for all
restore_scopes = ['FaceResNet']

# Weight decay for model variables
weight_decay = 5e-4

# Keep probability for dropouts
keep_prob = 1.0



####### LOSS FUNCTION #######

# Loss functions and their parameters
losses = {
    # 'softmax': {},
    # 'cosine': {'gamma': 'auto'},
    # 'angular': {'m': 4, 'lamb_min':5.0, 'lamb_max':1500.0},
    # 'am': {'gamma': 'auto', 'm': 10.0}
    'pair': {'gamma': 'auto', 'm': 0.5},
}

# Build batches with by concatenating pairs from different classes
# Set true only if you are using Max-margin Pair poss function.
use_pair_batch = True