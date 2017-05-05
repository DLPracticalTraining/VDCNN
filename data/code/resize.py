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
train_test = ['_train.txt', '_val.txt']
train_test_num = [0, 0]

for i in range(class_num):
    for tt in range(2):
        count = 0
        file_path_r = src_log_file_path + class_name[i] + train_test[tt]
        file_path_w = tar_log_file_path + str(i) + train_test[tt]
        fpr = open(file_path_r, 'r')
        fpw = open(file_path_w, 'w')

        flag = False
        for oneLine in fpr:
            datas = oneLine.split(' ')
            if datas[-1][0] == '1':
                train_test_num[tt] += 1
                
                if flag == False:
                    flag = True
                else:
                    fpw.write('\n')
                
                count = count + 1
                file_name = datas[0]
                image_path = src_path + file_name + '.jpg'
                image = Image.open(image_path)
                image_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))
                image_resized.save(tar_path + file_name + '.jpg')

                fpw.write(file_name)
                

        fpr.close()
        fpw.close()

        print class_name[i] + train_test[tt] + ' completed. Total number: %d' %count

print 'Total train number: %d'%train_test_num[0]
print 'Total test number: %d'%train_test_num[1]

