import tensorflow as tf
from gan import dcgan
from scipy.stats import multivariate_normal
import os 
import importlib
from data import cifar10, utilities
import numpy as np


def probability(variance, z_value, error):
	'''
	Very simple function which calculates the probability that the image
	is created by the generator
	Returns: 
		Pr_H the probability 
	'''
	zero_mean = np.zeros(3072)
	Pr_z = multivariate_normal.pdf(z_value, 0, 1)
	Pr_e = multivariate_normal.pdf(error, zero_mean, variance)
	return Pr_z , Pr_e


def anomaly_detected(threshold, Pr_H):
	'''
	Trivial boolean function
	'''
	return (Pr_H < threshold)


#########################################################################
####### We invert the function and try detect if it a normal or not #####
#########################################################################
graph = tf.Graph()
with graph.as_default(): 
	BATCH_SIZE = 1

	#Create one image to test inverting
	test_image = map((lambda inp: (inp[0]*2. -1., inp[1])), 
		             utilities.infinite_generator(cifar10.get_train(), BATCH_SIZE))
	inp, _ = next(test_image)
	#M_placeholder = tf.placeholder(tf.float32, shape=cifar10.get_shape_input(), name='M_input')
	M_placeholder = inp
	zmar = tf.summary.image('input_image', inp)
	#Create sample noise from random normal distribution
	z = tf.get_variable(name='z', shape=[BATCH_SIZE, 100], initializer=tf.random_normal_initializer())

	# Function g(z) zhere z is randomly generated
	g_z = dcgan.generator(z, is_training=True, name='generator')
	generator_visualisation = tf.cast(((g_z / 2.0) + 0.5) * 255.0, tf.uint8)
	sum_generator = tf.summary.image('summary/generator', generator_visualisation)

	img_summary = tf.summary.merge([sum_generator, zmar])
	with tf.name_scope('error'):
		error = M_placeholder - g_z
		# We set axis = None because norm(tensor, ord=ord) is equivalent to norm(reshape(tensor, [-1]), ord=ord)
		error_norm = tf.norm(error, ord=2, axis=None, keep_dims=False, name='L2Norm')
		summary_error = tf.summary.scalar('error_norm', error_norm)

	with tf.name_scope('Optimizing'):
		optimizer = tf.train.AdamOptimizer(0.01).minimize(error_norm, var_list=z)

	'''
	sv = tf.train.Supervisor(logdir='gan/train_logs/', save_summaries_secs=None, save_model_secs=None)
	tf.reset_default_graph()

	i = 0
	inp = next(test_image)

	with sv.managed_session() as sess:
		# Tensorboard loggings
		logwriter = tf.summary.FileWriter("gan/invert_logs/", sess.graph)

		while not sv.should_stop() and i < 100:
			sess.run(optimizer, summary_error, feed_dict={M_placeholder: inp})
			i +=1
	'''


tf.reset_default_graph()
with tf.Session(graph=graph) as sess:

	sess.run(tf.global_variables_initializer())
	ckpt = tf.train.get_checkpoint_state('gan/train_logs/')
	saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta', clear_devices=True)
	saver.restore(sess, ckpt.model_checkpoint_path)
	logwriter = tf.summary.FileWriter("gan/invert_logs/", sess.graph)
	#inp, _ = next(test_image)
	for i in range(100):
		#(_, s) = sess.run((optimizer, summary_error), feed_dict={M_placeholder: inp})
		(_, s) = sess.run((optimizer, summary_error))
		logwriter.add_summary(s, i)
		print('step %d: Patiente un peu poto!' % i)
		img = sess.run(img_summary)
		logwriter.add_summary(img, i)




	#### Zbel 	
	print(sess.run(z[0]))
######## Test ###########
	z_flat_array = sess.run(z[0])
	error_flat = tf.reshape(error, [-1])
	error_flat_array = sess.run(error_flat)
	variance = np.identity(3072)
	Pr_H = probability(variance, z_flat_array, error_flat_array)
	print(error_flat_array)
	print(Pr_H)
	#print(anomaly_detected(0.8, Pr_H))




			


