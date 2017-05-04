# VERY DEEP CONVOLUTIONAL NETWORKS(VGG, 2015)

This is a Tensorflow implementation of [VERY DEEP CONVOLUTIONAL NETWORKS(VGG, 2015)](https://arxiv.org/abs/1409.1556). We perform the testing with the classification task on [VOC2012](http://cvlab.postech.ac.kr/~mooyeol/pascal_voc_2012/). When processing the data, we use [CImg](https://github.com/dtschump/CImg) library to load the resized images.

## Data Processing

* download VOC2012 dataset and put it in `./data` directory
* `cd data/code`
* `python resize.py`
* `make`
* `./main.out`