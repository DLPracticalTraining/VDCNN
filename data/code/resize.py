import random
import re
import numpy as np
from PIL import Image

IMG_WIDTH = 224
IMG_HEIGHT = 224

src_path = '../VOC2012/JPEGImages/'
tar_path = '../images/'
src_log_file_path = '../VOC2012/ImageSets/Main/'
tar_log_file_path = '../logs/'

class_name = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', \
              'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', \
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', \
              'sheep', 'sofa', 'train', 'tvmonitor']

class_num = 20
train_test = ['train.txt', 'val.txt']
train_test_num = [0, 0]

train_text = ''
test_text = ''
for i in range(class_num):
    for tt in range(2):
        count = 0
        file_path_r = src_log_file_path + class_name[i] + '_' + train_test[tt]
        
        fpr = open(file_path_r, 'r')
        for oneLine in fpr:
            datas = oneLine.split(' ')
            if datas[-1][0] == '1':
                train_test_num[tt] += 1
                count = count + 1
                file_name = datas[0]
                if tt == 0:
                    train_text = train_text + file_name + ' ' + str(i) + '\n'
                else:
                    test_text = test_text + file_name + ' ' + str(i) + '\n'
                image_path = src_path + file_name + '.jpg'
                image = Image.open(image_path)
                image_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))
                image_resized.save(tar_path + file_name + '.jpg')

        fpr.close()

        print class_name[i] + train_test[tt] + ' completed. Total number: %d' %count

file_path_train = tar_log_file_path + train_test[0]
file_path_test = tar_log_file_path + train_test[1]
fp_train = open(file_path_train, 'w')
fp_test = open(file_path_test, 'w')

fp_train.write(str(train_test_num[0]) + '\n' + train_text)
fp_test.write(str(train_test_num[1]) + '\n' + test_text)

fp_train.close()
fp_test.close()

print 'Total train number: %d'%train_test_num[0]
print 'Total test number: %d'%train_test_num[1]

