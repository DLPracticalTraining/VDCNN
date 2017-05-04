import tensorflow as tf
import numpy as np
import argparse
from VOC2012 import *
from PIL import Image

def save_img(arr, path):
    if arr.shape[2] == 1:
        arr = arr.reshape(arr.shape[0], arr.shape[1])
    arr = arr * 255
    img = Image.fromarray(arr.astype(np.uint8))
    img.save(path)

data = VOC2012('../data', 10, 10)
batch_X, batch_y = data.train.next_batch(10)
# print batch_X
print batch_y

for i in range(10):
    save_img(batch_X[i, :, :, :], './' + str(i) + '.bmp')