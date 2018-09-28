% --------------------------------------------------------
% Copyright (c) Yichun Shi
% 
% The script is based on the face alignment code of SphereFace
% (code: https://github.com/wy1iu/sphereface/tree/master/preprocess/code)
%
% Intro:
% This script is used to detect the faces in training & testing datasets (CASIA & LFW).
% Face and facial landmark detection are performed by MTCNN 
% (paper: http://kpzhang93.github.io/papers/spl.pdf, 
%  code: https://github.com/kpzhang93/MTCNN_face_detection_alignment).
%
% Note:
% Please make sure
% (a) the dataset is structured as `dataset/idnetity/image`, e.g. `casia/id/001.jpg`
% (b) the folder name and image format (bmp, png, etc.) are correctly specified.
% (c) a list of absolute paths of all images in the dataset is generated for input.
% --------------------------------------------------------

clear;clc;close all;

%list of images
imglist = importdata('/path/to/input/imagelist.txt');
output_dir = '/path/to/output/dataset';

%% mtcnn settings
minSize   = 20;
factor    = 0.85;
threshold = [0.6 0.7 0.9];
imgSize     = [112, 96];
coord5point = [30.2946, 51.6963;
               65.5318, 51.5014;
               48.0252, 71.7366;
               33.5493, 92.3655;
               62.7299, 92.2041];

%% add toolbox paths
matCaffe       = '/path/to/caffe/matlab/';
pdollarToolbox = '/path/to/toolbox';
MTCNN          = '/path/to/mtcnn/code/codes/MTCNNv1';
addpath(genpath(matCaffe));
addpath(genpath(pdollarToolbox));
addpath(genpath(MTCNN));

%% caffe settings
gpu = 1;
if gpu
   gpu_id = 0;
   caffe.set_mode_gpu();
   caffe.set_device(gpu_id);
else
   caffe.set_mode_cpu();
end
caffe.reset_all();
modelPath = '/path/to/mtcnn/code/codes/MTCNNv1/model';
PNet = caffe.Net(fullfile(modelPath, 'det1.prototxt'), ...
                 fullfile(modelPath, 'det1.caffemodel'), 'test');
RNet = caffe.Net(fullfile(modelPath, 'det2.prototxt'), ...
                 fullfile(modelPath, 'det2.caffemodel'), 'test');
ONet = caffe.Net(fullfile(modelPath, 'det3.prototxt'), ...
                 fullfile(modelPath, 'det3.caffemodel'), 'test');


%% face and facial landmark detection
count = 0;
count_face = 0;
for i = 1:length(imglist)
    if mod(i, 100) == 0
        fprintf('detecting the %dth image...\n', i);
    end
    % load image
    img = imread(imglist{i});
    if size(img, 3) == 1
        img = repmat(img, [1,1,3]);
    end
    % detection
    [bboxes, landmarks] = detect_face(img, minSize, PNet, RNet, ONet, threshold, false, factor);

    if size(bboxes, 1) >= 1
        count = count + 1;
        count_face = count_face + size(bboxes, 1);
    end
    if size(bboxes, 1) > 1
        % pick the face closed to the center
        center   = size(img) / 2;
        distance = sum(bsxfun(@minus, [mean(bboxes(:, [2, 4]), 2), ...
                                       mean(bboxes(:, [1, 3]), 2)], center(1:2)).^2, 2);
        [~, Ix]  = min(distance);

        bboxes(:, 3:4) = bboxes(:, 3:4) - bboxes(:, 1:2);
        bboxes = bboxes(Ix,:);
        landmark = landmarks(:, Ix);
    else
        bbox =  bboxes(1, :);
        bbox(3:4) = bbox(3:4) - bbox(1:2);
        landmark = landmarks(:, 1);
    end
    
    % Align images and save
    landmark = double(reshape(landmark, [5 2]));
    transf   = cp2tform(landmark, coord5point, 'similarity');
    cropImg  = imtransform(img, transf, 'XData', [1 imgSize(2)],...
                                        'YData', [1 imgSize(1)], 'Size', imgSize);

    % save image
    [sPathStr, name, ext] = fileparts(imglist{i});
    clsName = strsplit(imglist{i}, '/');
    clsName = clsName{end-1};
    tPathStr = [output_dir '/' clsName];
    if ~exist(tPathStr, 'dir')
       mkdir(tPathStr)
    end
    tPathFull = fullfile(tPathStr, [name ext]);
    imwrite(cropImg, tPathFull, 'jpg');
end

disp('Finish Alignment.')




fprintf('%d out of %d images aligned, %d faces in all found.\n', count, length(imglist), count_face);

