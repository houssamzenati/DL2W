# Train DCGAN model on CIFAR-10 data
# Written by Kingsley Kuan

import tensorflow as tf
from data import cifar10, utilities
from . import dcgan
import logging
import numpy as np
from scipy.stats import multivariate_normal, norm, mvn
from sys import maxsize

logger = logging.getLogger("gan.test")

def probability(z_norm, error_norm, variance_error):
	'''
	Very simple function which calculates the probability that the image
	is created by the generator
	Returns: 
		Pr_H the probability 
	'''
	### Defauts parameters for z and error
	mean_z = 0
	variance_z = 1
	mean_error = 0

	### Actual computations of probabilities CDF

	### For z we focus on determining Z > |z|
	Pr_z_ = norm.cdf(z_norm, loc=mean_z, scale=variance_z)
	Pr_z = 2 * (1 - Pr_z_)  

	### For error we focus on determining Error < |error| 
	Pr_e_ = norm.cdf(error_norm, loc=mean_error, scale=variance_error)
	Pr_e = 2 * Pr_e_ - 1  
	### we take P(Error < |error|) = P(Error < error) - P(Error < -error)
	###                         = P(Error < error) - (1 - P(Error < error))

	### Final probability
	Pr_H = Pr_z * Pr_e

	return Pr_z, Pr_z_, Pr_e, Pr_e_, Pr_H


def anomaly_detected(threshold, Pr_H):
	'''
	Trivial boolean function
	'''
	return (Pr_H < threshold)


with tf.Session() as sess:

	##############################################################################################

	### Loading the previous trained generator
	train_dir = './gan/train_logs'
	ckpt = tf.train.latest_checkpoint(train_dir)
	filename = ".".join([ckpt, 'meta'])
	logger.info("Trained generator model from: {} is loading ...".format(filename))
	saver = tf.train.import_meta_graph(filename)
	saver.restore(sess, ckpt)
	tensors_to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')

	### Creating z variable
	batch_size = 1
	z = tf.get_variable(name='z', shape=[batch_size, 100], initializer=tf.random_normal_initializer())

	# Generate image of z with generator
	g_z = dcgan.generator(z, is_training=True, name='generator_z')

	# Add summaries to visualise output images 
	g_visualisation = tf.cast(((g_z / 2.0) + 0.5) * 255.0, tf.uint8)
	s_generator = tf.summary.image('summary/generator_z', g_visualisation)

	#Create one image to test inverting
	data_generator = map((lambda inp: (inp[0]*2. - 1., inp[1])), utilities.infinite_generator(cifar10.get_train(), batch_size))
	### Do it as many times as you want
	inpu, _ = next(data_generator)
	inpu, _ = next(data_generator)
	inpu, _ = next(data_generator)
	### in order to manually change the input image

	##############################################################################################################
	#############TEST#################
	random_z = tf.get_variable(name='random_z', shape=[batch_size, 100], initializer=tf.random_normal_initializer())

	# Generate images with generator
	inpu = dcgan.generator(random_z, is_training=True, name='generator_inp')

	# Add summaries to visualise output images and losses
	inpu_image = tf.cast(((inpu / 2.0) + 0.5) * 255.0, tf.uint8)

	###################################################################################################################################
	#Summary of the input image
	s_inp = tf.summary.image('input_image', inpu_image)

	# Merge summaries
	img_summary = tf.summary.merge([s_generator, s_inp])

	### Determining the error and setting its summary
	with tf.name_scope('Error'):
	  error = inpu - g_z
	  # We set axis = None because norm(tensor, ord=ord) is equivalent to norm(reshape(tensor, [-1]), ord=ord)
	  error_norm = tf.norm(error, ord=2, axis=None, keep_dims=False, name='L2Norm')
	  summary_error = tf.summary.scalar('error_norm', error_norm)

	  error_z = z - random_z
	  error_z_norm = tf.norm(error_z, ord=2, axis=None, keep_dims=False, name='L2Norm_error_z')
	  summary_error_z = tf.summary.scalar('error_z_norm', error_z_norm)

	with tf.name_scope('z_Norms'):
	  var_z_norm = tf.norm(z, ord=2, axis=None, keep_dims=False, name='L2Norm_var')
	  summary_z_var = tf.summary.scalar('variable_z_norm', var_z_norm)
	  ran_z_norm = tf.norm(random_z, ord=2, axis=None, keep_dims=False, name='L2Norm_ran')
	  summary_z_ran = tf.summary.scalar('random_z_norm', ran_z_norm)

	norm_summary = tf.summary.merge([summary_error, summary_z_var, summary_z_ran, summary_error_z])

	### Optimizer
	with tf.name_scope('Optimizing'):
	  optimizer = tf.train.AdamOptimizer(0.001).minimize(error_norm, var_list=z, name='optimizer')

	### Initializing generator tensors with loaded trained model
	graph = tf.get_default_graph()
	for tensor in tensors_to_restore:
		graph.get_tensor_by_name(tensor.name)

	variables_to_initialize = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator_z')

	for index, variable in enumerate(variables_to_initialize):
		sess.run(tf.assign(variable, tensors_to_restore[index]))

	variables_to_initialize2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator_inp')

	for index, variable in enumerate(variables_to_initialize2):
		sess.run(tf.assign(variable, tensors_to_restore[index]))

	### Initializing other variables 
	for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
		if variable not in variables_to_initialize:
			if variable not in variables_to_initialize2:
				sess.run(variable.initializer)


	### Check if nothing remains
	get_uninitialized = tf.report_uninitialized_variables()
	logger.info("Variables which are not initialized: {}".format(sess.run(get_uninitialized)))
	
	##############################################################################################

	logwriter = tf.summary.FileWriter("gan/invert_logs/", sess.graph)
	for batch in range(10001):
		if batch % 100 == 0:
			logger.info('Step {}: Minimizing the error...'.format(batch))

		(_, s) = sess.run((optimizer, norm_summary))
		logwriter.add_summary(s, batch)
		img = sess.run(img_summary)
		logwriter.add_summary(img, batch)

	logger.info('Computing probabilities...')


	z_norm = sess.run(var_z_norm)
	print(z_norm)
	error_norm = sess.run(error_norm)
	print(error_norm)
	Pr_z, Pr_z_, Pr_e, Pr_e_, Pr_H = probability(z_norm, error_norm, variance_error=5)

	### Primary tests
	print(Pr_z_, Pr_z, Pr_e_, Pr_e)
	print(Pr_H)
	print(anomaly_detected(threshold=0.8, Pr_H=Pr_H))