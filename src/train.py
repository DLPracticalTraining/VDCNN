from datetime import datetime
import os
import random
import sys
import time

import tensorflow as tf
import numpy as np

import vgg16
from VOC2012 import *

class Config():
	batch_size = 1
	img_height = 224
	img_width = 224
	num_channel = 3
	num_classes = 20
	num_images_train = 8331
	num_images_test = 8351
	wd = 5e-4
	stddev = 5e-2
	moving_average_decay = 0.999
	initialize = True

	# checkpoint path and filename
	logdir = "../log/train_log"
	params_dir = "../params"
	save_filename = "vgg16"

	checkpoint_iters = 100
	summary_iters = 10
	validate_iters = 20

def one_hot(batch_y, num_classes):
	y_ = np.zeros((batch_y.shape[0], num_classes))
	y_[np.arange(batch_y.shape[0]), batch_y] = 1
	return y_

def training(learn_rate = 0.01, num_epochs =20, save_model = False, debug = False):
	# assert len(train_x.shape) == 4
	# [num_images, img_height, img_width, num_channel] = train_x.shape
	# num_classes = labels.shape[-1]

	config = Config()
	# config.num_classes = num_classes

	with tf.Graph().as_default():

		model = vgg16.VGG16(config)

		voc2012 = VOC2012('../data', config.batch_size, config.batch_size)

		predicts = model.building(True)

		# loss function
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = model.labels, logits = predicts)
		loss = tf.reduce_mean(cross_entropy)
		model.loss_summary(loss)

		# # optimizer with decayed learning rate
		# global_step = tf.Variable(0, trainable=False)
		# learning_rate = tf.train.exponential_decay(learn_rate, global_step, num_steps*num_epochs, 0.1, staircase=True)
		# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

		trainer = tf.train.RMSPropOptimizer(1e-3)
		gradients = trainer.compute_gradients(loss)
		clipped_gradients = [(tf.clip_by_value(_[0], -1.0, 1.0), _[1]) for _ in gradients]
		optimizer = trainer.apply_gradients(clipped_gradients)

		# prediction for the training data
		predicts_result = tf.nn.softmax(predicts)

		# Initializing operation
		init_op = tf.global_variables_initializer()

		saver = tf.train.Saver(max_to_keep=100)

		sess_config = tf.ConfigProto()
		with tf.Session(config=sess_config) as sess:
			# initialize parameters or restore from previous model
			if not os.path.exists(config.params_dir):
				os.makedirs(config.params_dir)
				print "Initializing Network..."
				sess.run(init_op)
			else:
				ckpt = tf.train.get_checkpoint_state(config.params_dir)
				if ckpt and ckpt.model_checkpoint_path:
					print "Model Restoring..."
					model.restore(sess, saver, ckpt.model_checkpoint_path)
				else:
					print "Initializing Network..."
					sess.run(init_op)


			merged = tf.summary.merge_all()
			logdir = os.path.join(config.logdir,
				datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

			writer = tf.summary.FileWriter(logdir, sess.graph)

			
			print 'Training...'
			for epoch in range(num_epochs):
				imgs, labels = voc2012.train.next_batch(config.batch_size)
				labels = one_hot(labels, 20)

				feed_dict = {
					model.imgs: imgs,
					model.labels: labels
					}

				_, l, predictions, summary = sess.run([optimizer, loss, predicts_result, merged], feed_dict = feed_dict)	
				print "Epoch %d: Loss = %0.6f" % (epoch, l)

				# write summary
				# tmp_global_step = model.global_step.eval()
				writer.add_summary(summary, epoch)
				# save checkpoint
				if epoch % 10 == 0:
					tmp_global_step = model.global_step.eval()
					model.save(sess, saver, config.save_filename, tmp_global_step)

			print 'Testing...'
			test_loss = 0.0
			test_accuracy = 0.0
			batches = config.num_images_test / config.batch_size + 1
			for i in range(batches):			
				valid_x, valid_y = voc2012.test.next_batch(config.batch_size)
				valid_y = one_hot(valid_y, 20)
				
				feed_dict = {
					model.imgs: valid_x,
					model.labels: valid_y
					}

				l, predictions = sess.run([loss, predicts_result], feed_dict = feed_dict)
				test_loss += l
				print 'Batch Accuracy = %.6f%%' % accuracy(predictions, valid_y)
				test_accuracy += accuracy(predictions, valid_y)
			print 'Total Accuracy = %.6f%%' % test_accuracy / batches

# predictions/labels is a 2-D matrix [num_images, num_classes]
def accuracy(predictions, labels):
	return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

if __name__ == "__main__":
	# batch = 64
	# voc2012 = VOC2012('../data', batch, batch)
	
	# # imgs shape  (128, 224, 224, 3)
	# # labels shape (128, 20)
	# imgs, labels = voc2012.train.next_batch(batch)
	# labels = one_hot(labels, 20)
	
	# test_imgs, test_labels = voc2012.test.next_batch(batch)
	# test_labels = one_hot(test_labels, 20)

	# training(imgs, labels, test_imgs, test_labels)
	training()