import tensorflow as tf
import numpy as np
from datetime import datetime
import cv2
import os
import re
import random
import math
import sys
import time

#import vgg19_trainable as vgg19
import vgg16_trainable as vgg16
from VOC2012 import *

class Config():
  batch_size = 32
  steps = "-1"
  gpu = '/gpu:0'

  # checkpoint path and filename
  logdir = "../log/train_log/"
  params_dir = "../params/"
  # load_filename = "vgg16" + '-' + steps
  save_filename = "vgg16"

  # iterations config
  max_iteration = 1000
  summary_iters = 100


def one_hot(batch_y, num_classes):
  y_ = np.zeros((batch_y.shape[0], num_classes))
  y_[np.arange(batch_y.shape[0]), batch_y] = 1
  return y_


def accuracy(predictions, labels):
  # correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
  # return 100.0 * tf.reduce_mean(tf.cast(correct_prediction, "float"))
  return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


def run_model():
    config = Config()
    sess = tf.Session()

    images = tf.placeholder(tf.float32, [config.batch_size, 224, 224, 3])
    true_out = tf.placeholder(tf.float32, [config.batch_size, 20])
    train_mode = tf.placeholder(tf.bool)

    vgg = vgg16.Vgg16('../pre-vgg16.npy')
    vgg.build(images, train_mode)
    voc2012 = VOC2012('../data', config.batch_size, config.batch_size)

    # loss function
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=true_out, logits=vgg.fc9)
    loss = tf.reduce_mean(cross_entropy)
    vgg.loss_summary(loss)

    trainer = tf.train.RMSPropOptimizer(0.0001)
    gradients = trainer.compute_gradients(loss)
    clipped_gradients = [(tf.clip_by_value(_[0], -1.0, 1.0), _[1]) for _ in gradients]
    optimizer = trainer.apply_gradients(clipped_gradients)

    sess.run(tf.global_variables_initializer())

    if not os.path.exists(config.params_dir):
        os.makedirs(config.params_dir)

    merged = tf.summary.merge_all()
    logdir = os.path.join(config.logdir,
      datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    writer = tf.summary.FileWriter(logdir, sess.graph)

    # test classification
    # prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
    # utils.print_prob(prob[0], './synset.txt')

    print "start training"
    for idx in xrange(config.max_iteration):
      imgs, labels = voc2012.train.next_batch(config.batch_size)
      labels = one_hot(labels, 20)
      # feed data into the model
      feed_dict = {
          images : imgs,
          true_out : labels,
          train_mode : True
      }

      # with tf.device(config.gpu):
        # run the training operation
      # cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
      # train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
      _, batch_loss, summary = sess.run([optimizer, loss, merged], feed_dict=feed_dict)
      print "Epoch %d: the batch_loss = %0.6f" % (idx, batch_loss)

      writer.add_summary(summary, idx)

      # if idx == 10:
      #   vgg.save_npy(sess, "./params/vgg16-"+str(idx)+".npy")
      if (idx+1)% config.summary_iters == 0:
        vgg.save_npy(sess, "./params/vgg16-"+str(idx+1)+".npy")

if __name__ == "__main__":
  run_model()
