# *DocFace*: Matching ID Document Photos to Selfies

By Yichun Shi and Anil K. Jain

<img src="https://raw.githubusercontent.com/seasonSH/DocFace/master/figs/docface.png" width="600px">

### Contents
0. [Introduction](#introduction)
0. [Citation](#citation)
0. [Requirements](#requirements)
0. [Usage](#usage)
0. [Models](#models)
0. [Results](#results)


### Introduction

This repository includes the tensorflow implementation of **DocFace**, which is a system proposed for matching ID photos and live face photos. DocFace is shown to siginificantly outperformn general face matchers on the ID-Selfie matching problem. We here give the example training code and pre-trained models in the paper. For the preprocessing part, we follow the repository of [SphereFace](http://openaccess.thecvf.com/content_cvpr_2017/papers/Liu_SphereFace_Deep_Hypersphere_CVPR_2017_paper.pdf) to align the face images using [MTCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment). The user can also use other methods for face alignment. Because the dataset used in the paper is private, we cannot publish it here. One can test the system on their own dataset.


### Citation

If you find **DocFace** helpful to your research, please cite:

	@InProceedings{shi_2018_docface,
	  title = {DocFace: Matching ID Document Photos to Selfies},
	  author = {Shi, Yichun and Jain, Anil K.},
	  booktitle = {arXiv:1805.02283},
	  year = {2018}
	}

### Requirements
1. Requirements for `Python3`
2. Requirements for `Tensorflow 1.2r` or newer versions.
3. Run `pip install -r requirements.txt` for other dependencies.

### Usage

#### Part 1: Preprocessing
##### Dataset Structure
Download the [Ms-Celeb-1M](https://www.msceleb.org/download/cropped) and [LFW](http://vis-www.cs.umass.edu/lfw/lfw.tgz) dataset for training and testing the base model. Other dataset such as CASIA-Webface can also be used for training. Because Ms-Celeb-1M is known to be a very noisy dataset, we use the [clean list](https://github.com/AlfredXiangWu/face_verification_experiment) provided by Wu et al. Arrange Ms-Celeb-1M dataset and LFW dataset as the following structure, where each subfolder represents a subject:

    Aaron_Eckhart
        Aaron_Eckhart_0001.jpg
    Aaron_Guiel
        Aaron_Guiel_0001.jpg
    Aaron_Patterson
        Aaron_Patterson_0001.jpg
    Aaron_Peirsol
        Aaron_Peirsol_0001.jpg
        Aaron_Peirsol_0002.jpg
        Aaron_Peirsol_0003.jpg
        Aaron_Peirsol_0004.jpg
    ...

For the ID-Selfie dataset, make sure each folder has only two images and is in such a structure:

    Subject1
        1.jpg
        2.jpg
    Subject2
        1.jpg
        2.jpg
    ...
Here "1.jpg" are the ID photos and "2.jpg" are the selfies.

##### Face Alignment
We align all the face images following the [SphereFace](http://openaccess.thecvf.com/content_cvpr_2017/papers/Liu_SphereFace_Deep_Hypersphere_CVPR_2017_paper.pdf). The user is recommended to use their code for face alignment. It is okay to use other face alignment methods, but make sure all the images are resized to 96 x 112. Users can also use an input size of 112 x 112 by changing the "image_size" in the configuration files.

#### Part 2: Train
**Note:** In this part, we assume you are in the directory **`$DOCFACE_ROOT/`**

##### Training the base model

1. Set up the dataset paths in `config/basemodel.py`:

	```Python
	# Training dataset path
	train_dataset_path = '/path/to/msceleb1m/dataset/folder'
	
	# Testing dataset path
	train_dataset_path = '/path/to/lfw/dataset/folder'
	```

2. Due to the memory cost, the user may need more than one GPUs to use a batch size of `256` on Ms-Celeb-1M. In particular, we used four GTX 1080 Ti GPUs. In such cases, change the following entry in `config/basemodel.py`: 

    ```Python
    # Number of GPUs
    num_gpus = 1
    ```
 
    The user can also use the pre-trained [base model](#models) we provide.
 
3. Run the following command in the terminal:

	```Shell
	python train_base.py config/basemodel.py
	```
    After training, a model folder will appear under`log/faceres/`. We will use it for fine-tuning. If the training code is run more than once, multiple folders will appear with time stamps as their names.
    
##### Fine-tuning on the ID-Selfie datasets

1. Set up the dataset paths and the pre-trained model path in `config/finetune.py`

	```Python
	# Training dataset path
	train_dataset_path = '/path/to/training/dataset/folder'
	
	# Testing dataset path
	train_dataset_path = '/path/to/testing/dataset/folder'
	
	...
	
	# The model folder from which to retore the parameters
    restore_model = '/path/to/the/pretrained/model/folder'
	```

2. Run the following command in the terminal:

	```Shell
	python train_sibling.py config/finetune.py
	```

#### Part 3: Feature Extraction
**Note:** In this part, we assume you are in the directory **`$DOCFACE_ROOT/`**

To extract features using a pre-trained model (either base network or sibling network), prepare a `.txt` file of image list. Then run the following command in terminal:

```Shell
python extract_features.py \
--model_dir /path/to/pretrained/model/dir
--image_list /path/to/imagelist.txt
--output /path/to/output.npy
```

Notice that when extracting features using a sibling network, we assume that the images are in the order of template, selfie, template, selfie ... One needs to change the code for other cases.

### Models

- BaseModel: [Google Drive](https://drive.google.com/file/d/1YIZXsvtxQ4HkwGUDqq3bSwZVIv9e338R/view?usp=sharing)

- Fine-tuned DocFace model: [Google Drive](https://drive.google.com/file/d/1GJHjapZo8HcQ6aKEpSOZeAJCd9uaO5j1/view?usp=sharing)


### Results
- Using our pre-trained base model, one should be able to achieve 99.67% on the standard LFW verification protocol and 99.60% on the [BLUFR](http://www.cbsr.ia.ac.cn/users/scliao/projects/blufr/) protocol. Similar results should be achieved by using our code to train the Face-ResNet on Ms-Celeb-1M.

- Using our private dataset, we see a significant improvement of performance on the ID-Selfie matching problem:
    
    <img src="https://raw.githubusercontent.com/seasonSH/DocFace/master/figs/table1.png" width="500px">
    <br>
    <br>
    <br>
    <img src="https://raw.githubusercontent.com/seasonSH/DocFace/master/figs/table2.png" width="500px">

### Contact

  Yichun Shi: shiyichu **at** msu **dot** edu
