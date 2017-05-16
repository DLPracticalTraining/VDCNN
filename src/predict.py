import tensorflow as tf
import numpy as np
import cv2
import os
import re
import random
import math

#import vgg19_trainable as vgg19
import vgg16_trainable as vgg16
from VOC2012 import *

class Config():
  batch_size = 1
  steps = "1200"
  gpu = '/gpu:0'

  # checkpoint path and filename
  logdir = "../log/train_log/"
  params_dir = "../params/"
  load_filename = "vgg16" + '-' + steps + ".npy"
  save_filename = "vgg16"


  # iterations config
  max_iteration = 200
  summary_iters = 100



# reader = read_data.Reader(config)
def one_hot(batch_y, num_classes):
  y_ = np.zeros((batch_y.shape[0], num_classes))
  y_[np.arange(batch_y.shape[0]), batch_y] = 1
  return y_

def accuracy(predictions, labels):
  print "the predicted nums: " , np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
  # correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
  # return 100.0 * tf.reduce_mean(tf.cast(correct_prediction, "float"))
  return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

def test_model():
    config = Config()
    sess = tf.Session()

    predicts = np.zeros([200,20])
    labels = np.zeros([200,20])

    images = tf.placeholder(tf.float32, [config.batch_size, 224, 224, 3])
    true_out = tf.placeholder(tf.float32, [config.batch_size, 20])
    train_mode = tf.placeholder(tf.bool)
    vgg = vgg16.Vgg16("../params/vgg16-900.npy")
    #vgg = vgg16.Vgg16('./params/vgg16-99.npy')
    #vgg = vgg19.Vgg19('./params/vgg19-1999.npy')
    vgg.build(images, train_mode)
    voc2012 = VOC2012('../data', config.batch_size, config.batch_size)

    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    # print(vgg.get_var_count())

    sess.run(tf.global_variables_initializer())

    count = 0.0
    for idx in xrange(config.max_iteration):
      imgs, label = voc2012.test.next_batch(config.batch_size)
      label = one_hot(label, 20)
      labels[idx] = label

      # feed data into the model
      feed_dict = {
          images : imgs,
          true_out : label,
          train_mode : False
      }

      # with tf.device(config.gpu):
        # run the training operation
#        cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
#        train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
#        sess.run(train, feed_dict=feed_dict)

#     if (idx+1)% config.summary_iters == 0:
#        print "ecoch:",idx+1, "cost:", cost
#        vgg.save_npy(sess, "./params/vgg19-"+str(idx)+".npy")
      prob = sess.run(vgg.prob, feed_dict={images:imgs, train_mode:False})
      predicts[idx] = prob
      if (idx+1) % 100 == 0:
        print "Epoch: " , (idx+1)
        print count
        #utils.print_prob(prob[0], './synset.txt')
      temp = prob[:].argmax()
      if label[0, temp] > 0:
        count += 1.0
    print "the predicted num = %d" % count
    print "Total Accuracy = %.6f" % (count / 200)
    print 'Total Accuracy = %.6f%%' % accuracy(predicts, labels)
    
if __name__ == "__main__":
  test_model()
