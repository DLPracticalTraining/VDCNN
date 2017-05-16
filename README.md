# VERY DEEP CONVOLUTIONAL NETWORKS(VGG, 2015)

This is a Tensorflow implementation of [VERY DEEP CONVOLUTIONAL NETWORKS(VGG, 2015)](https://arxiv.org/abs/1409.1556). We perform the testing with the classification task on [VOC2012](http://cvlab.postech.ac.kr/~mooyeol/pascal_voc_2012/). When processing the data, we use [CImg](https://github.com/dtschump/CImg) library to load the resized images.

## Data Processing

* Download VOC2012 dataset, rename the dataset folder as `VOC2012` and put it in `./data` directory
* Put the source code of CImg library (ie. [`CImg.h`](https://github.com/dtschump/CImg/blob/master/CImg.h)) in `./data/code` directory
* `cd data/code`
* `python resize.py`
* `make`
* `./main.out`

## Training

* `cd src`
* `python train.py`

## Accessing Training Data of Steps

* `tensorboard --logdir=./log`