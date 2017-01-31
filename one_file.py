import tensorflow as tf
from tensorflow.python.platform import gfile
import glob
import os
import random
import sys
import numpy as np
from skimage import io
from skimage.transform import resize
import pickle
import time

video_shape = [32,64,64,3]
global Options
Options = None

#------------------------------utils.py-----------------------------------
class Dataset(object):
	def __init__(self, 
				 flags):
		global Options
		Options = flags
		print("Loading paths of video data")

		self._calc_mean = Options.calc_mean
		self._train = self._load_data(Options.dataset)

		# Load mean . Shape [32*128, 128, 3]
		self._mean = io.imread(Options.mean_path)
		# Reshape to [32, 128, 128, 3]
		self._mean = np.reshape(self._mean, [-1,128,128,3])
		new_mean = np.zeros(shape=video_shape)
		#Now reshape each [128,128,3] image to [64,64,3] shaped image
		for i, im in enumerate(self._mean):
			new_mean[i] = resize(im,(64,64,3),order=3)
		self._mean = new_mean
		print("Read dataset")

	def _load_data(self, 
				   start_dir):
		'''Loads paths of all images contained under the directory 'start_dir'
		Input:
			start_dir: The path to the data directory where all images are stored
		Output:
			List of paths to images
		'''
		data = []
		begin = time.time()
		try:
			with open (Options.data_list, 'rb') as fp:
				data = pickle.load(fp)
				print("Data loaded from cached list")
		except:	
			pattern = "*.jpg"
			for dr,_,_ in os.walk(start_dir):
				data.extend(glob.glob(os.path.join(dr,pattern)))

			random.shuffle(data)

			with open(Options.data_list, 'wb') as fp:
				pickle.dump(data, fp)
				print("Saving data list to %s"%Options.data_list)
		print("Time taken to load data = %.2f seconds"%(time.time()-begin))

		if self._calc_mean or not os.path.exists(Options.mean_path):

			print("Checking integrity of files and calculationg mean. This will take several minutes and needs to be done only once.")

			mean_img = np.zeros(shape=[32,128,128,3], dtype=np.float32)
			count = 0
			invalid_count = 0
			for num,a in enumerate(data):
				try:	# Try to read an image
					sys.stdout.write("\r%i/%i files processed"%(num,len(data)))
					sys.stdout.flush()

					ims = io.imread(a)

					try:	# Now calculate mean
						ims = ims.reshape(-1,128,128,3)
						
						# We need 32 frames
						if ims.shape[0] > video_shape[0]: # 32
							ims = ims[:video_shape[0], :, :, :]
						elif ims.shape[0] < video_shape[0]: #32
							# Copy the last frame till 32
							frames = ims.shape[0]
							repeats = [1 for i in range(frames)]
							repeats[-1] = video_shape[0] - frames + 1
							ims = np.repeat(ims, repeats, axis=0)

						# The running average
						mean_img = mean_img*(float(count)/(count+1)) + (ims)/(count+1.0)
						count += 1
					except:
						print("\nError in calculating mean")
				except ValueError:
					print("\nDelete the file(s) %s. Restart the program after deleting above mentioned files"%a)
					invalid_count += 1

			print("\n%i / %i files corrupted"%(invalid_count,len(data)))
			print("Saving mean")
			mean_img/=255.0
			mean_img = np.reshape(mean_img,[-1,128,3])

			# Save the mean
			io.imsave(Options.mean_path, mean_img)
		
		return data

	def train_iter(self):
		return self._get_iter(self._train)

	def _get_iter(self,
				  data):
		n = len(data)
		num_batches = n // Options.batch_size
		for i in range(num_batches-1): #Ignore the last batch as this might create some problem regarding batch_size

			ind1 = i*Options.batch_size
			ind2 = min(n, (i + 1) * Options.batch_size)

			batch_paths = data[ind1:ind2]

			batch = np.zeros(shape=[ind2-ind1]+video_shape) # Prefixing with batch_size

			for video_num,path in enumerate(batch_paths):
				# Read image [a*128, 128, 3]
				ims = io.imread(path)
				ims = ims.reshape(-1,128,128,3)

				# We have to make sure a=32
				if ims.shape[0] > video_shape[0]: # 32
					# Use only 1st 32 frames
					ims = ims[:video_shape[0], :, :, :]
				elif ims.shape[0] < video_shape[0]: #32
					# Copy the last frame till 32
					frames = ims.shape[0]
					repeats = [1 for i in range(frames)]
					repeats[-1] = video_shape[0] - frames + 1
					ims = np.repeat(ims, repeats, axis=0)

				new_ims = np.zeros(shape=video_shape)
				for i, im in enumerate(ims):
					new_ims[i] = resize(im,(64,64,3),order=3)

				batch[video_num] = new_ims - self._mean

			yield batch


def save_samples(samples,
				 epoch,
				 counter):
	"""Saves sample videos from generator
	Input:
		samples: sample_size number of videos
		epoch: the epoch in which this was created
		counter: the batch number
	"""
	print("Saving samples")
	if not os.path.exists(Options.sample_path):
		os.makedirs(Options.sample_path)

	folder = 'Epoch_%s_Counter_%s'%(str(epoch),str(counter))
	folder = os.path.join(Options.sample_path,folder)
	if not os.path.exists(folder):
		os.makedirs(folder)

	# Load the mean
	_mean = io.imread(Options.mean_path)
	_mean = np.reshape(_mean, [-1,128,128,3])
	new_mean = np.zeros(shape=video_shape)
	for i, im in enumerate(_mean):
		new_mean[i] = resize(im,(64,64,3),order=3)
	_mean = new_mean	
	# samples shape is [sample_size]+video_shape = [ss, 32, 64, 64, 3]
	for sample_num in range(len(samples)):
		sample = samples[sample_num] + _mean
		sample /= 2.0
		sample = np.reshape(sample, [-1,64,3])
		name = str(sample_num)+'.jpg'
		name = os.path.join(folder,name)
		io.imsave(name, sample)
#--------------------------------------------------------------------------



#------------------------------modules.py----------------------------------

try:
	image_summary = tf.summary.image
	scalar_summary = tf.summary.scalar
	histogram_summary = tf.summary.histogram
	merge_summary = tf.summary.merge
	SummaryWriter = tf.summary.FileWriter
except:
	image_summary = tf.image_summary
	scalar_summary = tf.scalar_summary
	histogram_summary = tf.histogram_summary
	merge_summary = tf.merge_summary
	SummaryWriter = tf.train.SummaryWriter


class batch_norm(object):
	def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
		with tf.variable_scope(name):
			self.epsilon  = epsilon
			self.momentum = momentum
			self.name = name

	def __call__(self, x, train=True):
		return tf.contrib.layers.batch_norm(x,
											decay=self.momentum, 
											updates_collections=None,
											epsilon=self.epsilon,
											scale=True,
											scope=self.name)


def conv2d(input_, output_dim, 
		   k_h=4, k_w=4, d_h=2, d_w=2, stddev=0.02,
		   name="conv2d"):
	with tf.variable_scope(name):
		w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
							initializer=tf.truncated_normal_initializer(stddev=stddev))
		conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

		biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
		conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

		return conv


def conv3d(input_, output_dim, 
		   k_d=4, k_h=4, k_w=4, d_d=2, d_h=2, d_w=2, stddev=0.02,
		   name="conv2d"):
	with tf.variable_scope(name):
		w = tf.get_variable('w', [k_d, k_h, k_w, input_.get_shape()[-1], output_dim],
							initializer=tf.truncated_normal_initializer(stddev=stddev))
		conv = tf.nn.conv3d(input_, w, strides=[1, d_d, d_h, d_w, 1], padding='SAME')

		biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
		conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

		return conv		

def deconv2d(input_, output_shape,
			 k_h=4, k_w=4, d_h=2, d_w=2, stddev=0.02,
			 name="deconv2d", with_w=False):
	with tf.variable_scope(name):
		# filter : [height, width, output_channels, in_channels]
		w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
							initializer=tf.random_normal_initializer(stddev=stddev))
		
		try:
			deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
								strides=[1, d_h, d_w, 1])

		# Support for verisons of TensorFlow before 0.7.0
		except AttributeError:
			deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
								strides=[1, d_h, d_w, 1])

		biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
		deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

		if with_w:
			return deconv, w, biases
		else:
			return deconv
	   

def deconv3d(input_, output_shape,
			 k_d=4, k_h=4, k_w=4, d_d=2, d_h=2, d_w=2, stddev=0.02,
			 name="deconv3d", with_w=False):
	with tf.variable_scope(name):
		# filter : [depth, height, width, output_channels, in_channels]
		w = tf.get_variable('w', [k_d, k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
							initializer=tf.random_normal_initializer(stddev=stddev))
		
		try:
			deconv = tf.nn.conv3d_transpose(input_, w, output_shape=output_shape,
								strides=[1, d_d, d_h, d_w, 1])

		# Support for verisons of TensorFlow before 0.7.0
		except AttributeError:
			deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
								strides=[1, d_d, d_h, d_w, 1])

		biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
		deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

		if with_w:
			return deconv, w, biases
		else:
			return deconv
	   

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
	shape = input_.get_shape().as_list()

	with tf.variable_scope(scope or "Linear"):
		matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
								 tf.random_normal_initializer(stddev=stddev))
		bias = tf.get_variable("bias", [output_size],
			initializer=tf.constant_initializer(bias_start))
		if with_w:
			return tf.matmul(input_, matrix) + bias, matrix, bias
		else:
			return tf.matmul(input_, matrix) + bias
#-------------------------------------------------------------------------


#-----------------------------nnet.py-------------------------------------

class videoGan():

	def __init__(self, 
				 flags, 
				 batch_size, 
				 sample_size,
				 z_dim,
				 gf_dim,
				 df_dim,
				 c_dim,
				 mask_penalty):

		"""
		Input:
			flags: All options and hyperparameters
			batch_size: The size of batch. Should be specified before training.
			video_shape: Shape of videos. [32,64,64,3]
			batch_size: Batch size for training algorithm
			sample_size: Number of samples to be generated at once
			z_dim: Dimension of dim for Z. [100]
			gf_dim: Dimension of gen filters in first conv layer. [64]
			df_dim: Dimension of discrim filters in first conv layer. [64]
			c_dim: Dimension of image color. For grayscale input, set to 1. [3]
			mask_penalty: Lambda for L1 regularizer of mask
		"""
		global Options
		Options = flags
		self.batch_size = batch_size
		self.video_shape = [32,64,64,3]
		self.sample_size = sample_size

		self.z_dim = z_dim

		self.gf_dim = gf_dim
		self.df_dim = df_dim

		self.c_dim = c_dim
		self.mask_penalty = mask_penalty

		# Batch_normalization for discriminator
		self.d_bn1 = batch_norm(name='d_bn1')
		self.d_bn2 = batch_norm(name='d_bn2')
		self.d_bn3 = batch_norm(name='d_bn3')

		# Batch norms for static branch of generator
		self.g_sbn0 = batch_norm(name='g_sbn0')
		self.g_sbn1 = batch_norm(name='g_sbn1')
		self.g_sbn2 = batch_norm(name='g_sbn2')
		self.g_sbn3 = batch_norm(name='g_sbn3')

		# Batch norms for video branch of generator
		self.g_vbn0 = batch_norm(name='g_vbn0')
		self.g_vbn1 = batch_norm(name='g_vbn1')
		self.g_vbn2 = batch_norm(name='g_vbn2')
		self.g_vbn3 = batch_norm(name='g_vbn3')

		# Define the placeholders
		# videos - Discriminator requires as input real videos to differentiate from fake generated ones
		self.videos = tf.placeholder(
			dtype = tf.float32, 
			shape = [self.batch_size] + self.video_shape,
			name = 'real_videos')
		# sample_videos - To evaluate generator performance, we frequently extract sample videos
		self.sample_videos= tf.placeholder(
			dtype = tf.float32, 
			shape = [self.sample_size] + self.video_shape,
			name = 'sample_videos')
		# z - The random noise from where the genrator starts generating
		self.z = tf.placeholder(
			dtype = tf.float32, 
			shape = [None, self.z_dim],
			name = 'z')

		self.z_sum = histogram_summary("z", self.z)

		# generated videos, L1 regularizer penalty from generator
		self.G, self.g_loss_penalty = self.generator(self.z)
		# sigmoid of logits, logits for real videos from discriminator
		self.D, self.D_logits = self.discriminator(self.videos)

		# sample videos from sampler
		self.sampler = self.sampler(self.z)
		# sigmoid of logits, logits for fake videos from discriminator
		self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
		
		self.d_sum = histogram_summary("d", self.D)
		self.d__sum = histogram_summary("d_", self.D_)

		# For real videos, logits must be matched against 1s
		self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
		# For fake videos, logits must be matched against 0s
		self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
		# Generator loss is modelled
		self.g_loss_no_penalty = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

		self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
		self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
		
		# Total discriminator loss
		self.d_loss = self.d_loss_real + self.d_loss_fake

		self.g_loss_no_penalty_sum = scalar_summary("g_loss_no_penalty", self.g_loss_no_penalty)
		self.g_loss_penalty_sum = scalar_summary("g_loss_penalty", self.g_loss_penalty)

		# Total generator loss
		self.g_loss = self.mask_penalty*self.g_loss_penalty + self.g_loss_no_penalty

		self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
		self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

		t_vars = tf.trainable_variables()

		self.d_vars = [var for var in t_vars if 'd_' in var.name]
		self.g_vars = [var for var in t_vars if 'g_' in var.name]

		# Saver to checkpoint models
		self.saver = tf.train.Saver()

		print("Model defined")

		total_parameters = 0
		for var in tf.trainable_variables():
			print(var.name, var.get_shape())
			params = 1
			for dim in var.get_shape().as_list():
				params *= int(dim)
			total_parameters += params
		print("Total parameters = %i"%total_parameters)


	def discriminator(self, video, y=None, reuse=False):
		if reuse:
			tf.get_variable_scope().reuse_variables()
		
		# print("Printing discriminator output shapes")
		# print("video - shape = ", video.get_shape().as_list())  		# [bs, 32, 64, 64, 3]
		
		h0 = lrelu(conv3d(video, self.df_dim, name='d_h0_conv'))
		# print("video -> h0 - shape = ", h0.get_shape().as_list())		# [bs, 16, 32, 32, 64]
		
		h1 = lrelu(self.d_bn1(conv3d(h0, self.df_dim*2, name='d_h1_conv')))
		# print("h0 -> h1 - shape = ", h1.get_shape().as_list())		# [bs, 8, 16, 16, 128]
		
		h2 = lrelu(self.d_bn2(conv3d(h1, self.df_dim*4, name='d_h2_conv')))
		# print("h1 -> h2 - shape = ", h2.get_shape().as_list())		# [bs, 4, 8, 8, 256]
		
		h3 = lrelu(self.d_bn3(conv3d(h2, self.df_dim*8, name='d_h3_conv')))
		# print("h2 -> h3 - shape = ", h3.get_shape().as_list())		# [bs, 2, 4, 4, 512]
		
		h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
		# print("h3 -> h4 - shape = ", h4.get_shape().as_list())		# [bs, 1]

		return tf.nn.sigmoid(h4), h4


	def generator(self, z, y=None):
		s = 64
		s2, s4, s8, s16, s32 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32)
		
		# print("Inside generator")
		# print("z- shape = ", z.get_shape().as_list())                   # [None, 100]
		
		# s stands for static part
		self.sz_, self.sh0_w, self.sh0_b = linear(z, self.gf_dim*8*s16*s16, 'g_sh0_lin', with_w=True)
		# print("z -> sz_ - shape = ", self.sz_.get_shape().as_list())	# [None, 8192]
		
		self.sh0 = tf.reshape(self.sz_, [-1, s16, s16, self.gf_dim * 8])
		sh0 = tf.nn.relu(self.g_sbn0(self.sh0))
		# print("sz_ -> sh0 - shape = ", sh0.get_shape().as_list())		# [None, 4, 4, 512]
		
		self.sh1, self.sh1_w, self.sh1_b = deconv2d(sh0, 
			[self.batch_size, s8, s8, self.gf_dim*4], name='g_sh1', with_w=True)
		sh1 = tf.nn.relu(self.g_sbn1(self.sh1))
		# print("sh0 -> sh1 - shape = ", sh1.get_shape().as_list())		# [bs, 8, 8, 256]

		sh2, self.sh2_w, self.sh2_b = deconv2d(sh1,
			[self.batch_size, s4, s4, self.gf_dim*2], name='g_sh2', with_w=True)
		sh2 = tf.nn.relu(self.g_sbn2(sh2))
		# print("sh1 -> sh2 - shape = ", sh2.get_shape().as_list())		# [bs, 16, 16, 128]

		sh3, self.sh3_w, self.sh3_b = deconv2d(sh2,
			[self.batch_size, s2, s2, self.gf_dim*1], name='g_sh3', with_w=True)
		sh3 = tf.nn.relu(self.g_sbn3(sh3))
		# print("sh2 -> sh3 - shape = ", sh3.get_shape().as_list())		# [bs, 32, 32, 64]

		sh4, self.sh4_w, self.sh4_b = deconv2d(sh3,
			[self.batch_size, s, s, self.c_dim], name='g_sh4', with_w=True)
		# print("sh3 -> sh4 - shape = ", sh4.get_shape().as_list())		# [bs, 64, 64, 3]

		background = tf.nn.tanh(sh4)

		# Extending static part over time by replicating. Reshape is needed to increase dimension
		background = tf.tile(tf.reshape(background,[self.batch_size, 1, s, s, self.c_dim]), [1, s2, 1, 1, 1])
		# print("sh4-> background -> tile(background) - shape = ", background.get_shape().as_list())	# [bs, 32, 64, 64, 3]

		# v stands for video part
		self.vz_, self.vh0_w, self.vh0_b = linear(z, self.gf_dim*8*s32*s16*s16, 'g_vh0_lin', with_w=True)
		# print("z -> vz_ - shape = ", sh1.get_shape().as_list())			# [bs, 8, 8, 256]

		self.vh0 = tf.reshape(self.vz_, [-1, s32, s16, s16, self.gf_dim * 8])
		vh0 = tf.nn.relu(self.g_vbn0(self.vh0))
		# print("vz_ -> vh0 - shape = ", vh0.get_shape().as_list())		# [None, 2, 4, 4, 512]

		self.vh1, self.vh1_w, self.vh1_b = deconv3d(vh0, 
			[self.batch_size, s16, s8, s8, self.gf_dim*4], name='g_vh1', with_w=True)
		vh1 = tf.nn.relu(self.g_vbn1(self.vh1))
		# print("vh0 -> vh1 - shape = ", vh1.get_shape().as_list())		# [bs, 4, 8, 8, 256]

		vh2, self.vh2_w, self.vh2_b = deconv3d(vh1,
			[self.batch_size, s8, s4, s4, self.gf_dim*2], name='g_vh2', with_w=True)
		vh2 = tf.nn.relu(self.g_vbn2(vh2))
		# print("vh1 -> vh2 - shape = ", vh2.get_shape().as_list())		# [bs, 8, 16, 16, 128]

		vh3, self.vh3_w, self.vh3_b = deconv3d(vh2,
			[self.batch_size, s4, s2, s2, self.gf_dim*1], name='g_vh3', with_w=True)
		vh3 = tf.nn.relu(self.g_vbn3(vh3))
		# print("vh2 -> vh3 - shape = ", vh3.get_shape().as_list())		# [bs, 16, 32, 32, 64]

		mask_out, mask_out_w, mask_out_b = deconv3d(vh3,
			[self.batch_size, s2, s, s, 1], name='g_mask', with_w=True)

		mask_out = tf.nn.sigmoid(mask_out)
		# print("vh3 -> mask_out - shape = ", mask_out.get_shape().as_list())	# [bs, 32, 64, 64, 1]

		vh4, self.vh4_w, self.vh4_b = deconv3d(vh3,
			[self.batch_size, s2, s, s, self.c_dim], name='g_vh4', with_w=True)

		foreground = tf.nn.tanh(vh4)
		# print("vh3 -> vh4=foreground - shape = ", foreground.get_shape().as_list()) # [bs, 32, 64, 64, 3]

		# f = f*m
		foreground = tf.mul(foreground, mask_out)
		# b = b*(1-m)
		background = tf.mul(background, tf.sub(tf.constant([1.0]), mask_out))

		gen_video = tf.add(foreground, background)
		return gen_video, tf.reduce_mean(tf.reduce_sum(tf.abs(mask_out_w)))


	def sampler(self, z, y=None):
		""" Same as generator"""
		tf.get_variable_scope().reuse_variables()

		s = 64
		s2, s4, s8, s16, s32 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32)

		# s stands for static part
		self.sz_ = linear(z, self.gf_dim*8*s16*s16, 'g_sh0_lin')

		self.sh0 = tf.reshape(self.sz_, [-1, s16, s16, self.gf_dim * 8])
		sh0 = tf.nn.relu(self.g_sbn0(self.sh0))

		self.sh1 = deconv2d(sh0, 
			[self.batch_size, s8, s8, self.gf_dim*4], name='g_sh1')
		sh1 = tf.nn.relu(self.g_sbn1(self.sh1))

		sh2 = deconv2d(sh1,
			[self.batch_size, s4, s4, self.gf_dim*2], name='g_sh2')
		sh2 = tf.nn.relu(self.g_sbn2(sh2))

		sh3 = deconv2d(sh2,
			[self.batch_size, s2, s2, self.gf_dim*1], name='g_sh3')
		sh3 = tf.nn.relu(self.g_sbn3(sh3))

		sh4 = deconv2d(sh3,
			[self.batch_size, s, s, self.c_dim], name='g_sh4')

		background = tf.nn.tanh(sh4)

		# Extending static part over time by replicating. Reshape is needed to increase dimension
		background = tf.tile(tf.reshape(background,[self.sample_size, 1, s, s, self.c_dim]), [1, s2, 1, 1, 1])

		# v stands for video part
		self.vz_ = linear(z, self.gf_dim*8*s32*s16*s16, 'g_vh0_lin')

		self.vh0 = tf.reshape(self.vz_, [-1, s32, s16, s16, self.gf_dim * 8])
		vh0 = tf.nn.relu(self.g_vbn0(self.vh0))

		self.vh1 = deconv3d(vh0, 
			[self.sample_size, s16, s8, s8, self.gf_dim*4], name='g_vh1')
		vh1 = tf.nn.relu(self.g_vbn1(self.vh1))

		vh2 = deconv3d(vh1,
			[self.sample_size, s8, s4, s4, self.gf_dim*2], name='g_vh2')
		vh2 = tf.nn.relu(self.g_vbn2(vh2))

		vh3 = deconv3d(vh2,
			[self.sample_size, s4, s2, s2, self.gf_dim*1], name='g_vh3')
		vh3 = tf.nn.relu(self.g_vbn3(vh3))

		mask_out = deconv3d(vh3,
			[self.sample_size, s2, s, s, 1], name='g_mask')

		mask_out = tf.nn.sigmoid(mask_out)

		vh4 = deconv3d(vh3,
			[self.sample_size, s2, s, s, self.c_dim], name='g_vh4')

		foreground = tf.nn.tanh(vh4)

		# f = f*m
		foreground = tf.mul(foreground, mask_out)
		# b = b*(1-m)
		background = tf.mul(background, tf.sub(tf.constant([1.0]), mask_out))

		gen_video = tf.add(foreground, background)
		return gen_video
	 

	def prefix(self):
		prefix = "videogan_"
		prefix += "bs_%d_" % (Options.batch_size)
		prefix += "ss_%d_" % (Options.sample_size)
		prefix += "lrd_%.5f_" % (Options.lrate_d)
		prefix += "lrg_%.5f_" % (Options.lrate_g)
		prefix += "betad_%.3f_" % (Options.beta1_d)
		prefix += "betag_%.3f_" % (Options.beta1_g)
		return prefix

	def train(self, dataset):
		
		try:
			temp = set(tf.global_variables())
		except:
			temp = set(tf.all_variables())

		print("Creating optimizer")
		d_optim = tf.train.AdamOptimizer(Options.lrate_d, beta1=Options.beta1_d) \
						  .minimize(self.d_loss, var_list=self.d_vars)
		g_optim = tf.train.AdamOptimizer(Options.lrate_g, beta1=Options.beta1_g) \
						  .minimize(self.g_loss, var_list=self.g_vars)
		print("Optimizer created")

		with tf.Session() as sess:
			self.sess = sess
			try:
				print("Restoring from checkpoint")
				_dir = os.path.join(Options.checkpoint_dir, self.prefix())
				a = glob.glob(os.path.join(_dir, '*/'))
				a = [int(i.split('/')[-2]) for i in a]
				a.sort()
				_dir = os.path.join(_dir, str(a[-1]), 'model.ckpt')
				self.saver.restore(self.sess, _dir)
				print("Model restored")
				try:
					self.sess.run(tf.variables_initializer(set(tf.global_variables()) - temp))
				except:
					self.sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
			except:
				print("Initializing")
				try:
					tf.global_variables_initializer().run()
				except:
					tf.initialize_all_variables().run()
				print("Initialized")

			print("Merging summaries")
			self.g_sum = merge_summary([self.z_sum, self.d__sum,
										 self.d_loss_fake_sum, self.g_loss_sum])
			self.d_sum = merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
			self.writer = SummaryWriter("./logs", self.sess.graph)
			print("Summaries merged")

			sample_z = np.random.uniform(-1, 1, size=(self.sample_size , self.z_dim))

			counter = 0
			terrD_fake = 0.0
			terrD_real = 0.0
			terrG = 0.0

			s_begin = time.time()
			c_begin = time.time()
			p_begin = time.time()

			print("Starting training epoch")
			for epoch in range(Options.train_epochs):
				for sub_data in dataset.train_iter():

					batch_z = np.random.uniform(-1, 1, [Options.batch_size, self.z_dim]).astype(np.float32)

					# Update D network
					_, summary_str = self.sess.run([d_optim, self.d_sum],
						feed_dict={ self.videos: sub_data, self.z: batch_z })
					self.writer.add_summary(summary_str, counter)

					# Update G network
					_, summary_str = self.sess.run([g_optim, self.g_sum],
						feed_dict={ self.z: batch_z })
					self.writer.add_summary(summary_str, counter)
						
					errD_fake = self.d_loss_fake.eval({self.z: batch_z})
					errD_real = self.d_loss_real.eval({self.videos: sub_data})
					errG = self.g_loss.eval({self.z: batch_z})

					terrD_real += errD_real
					terrD_fake += errD_fake
					terrG += errG

					counter += 1

					if time.time() - p_begin > Options.print_time:
						print("Epoch: [%d], d_loss_fake: [%.6f]--[%.4f], d_loss_real: [%.6f]--[%.4f], g_loss: [%.6f]--[%.4f]"
						 	% (epoch, terrD_fake/counter, errD_fake, terrD_real/counter, errD_real, terrG/counter, errG))
						p_begin = time.time()

					if time.time() - s_begin > Options.sampler_time:
						samples, d_loss, g_loss = self.sess.run(
							[self.sampler, self.d_loss, self.g_loss],
							feed_dict={self.z: sample_z, self.videos: sub_data}
						)
						save_samples(samples, epoch, counter)
						print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
						s_begin = time.time()

					if time.time() - c_begin > Options.checkpoint_time:
						print("Checkpointing")
						if not os.path.exists(Options.checkpoint_dir):
							os.makedirs(Options.checkpoint_dir)
						_dir = os.path.join(Options.checkpoint_dir, self.prefix())
						self.save(_dir, epoch, self.sess)
						c_begin = time.time()
						
						counter = 0
						terrD_fake = 0.0
						terrD_real = 0.0
						terrG = 0.0


	def save(self,
			 _dir,
			 epoch,
			 sess):
		'''Checkpoints a tensorflow model
		Input:
			_dir: Directory to save checkpoint in
			epoch: A parameter needed for saver
			sess: Current session to be saved
		'''
		if not os.path.exists(_dir):
			os.makedirs(_dir)
		_sub_dir = os.path.join(_dir, str(epoch))
		if not os.path.exists(_sub_dir):
			os.makedirs(_sub_dir)
		self.saver.save(sess, os.path.join(_sub_dir, 'model.ckpt'))
		print("Checkpointed")

#-------------------------------------------------------------------------



#----------------------------main.py-----------------------------------
FLAGS = tf.app.flags.FLAGS

# Data
tf.app.flags.DEFINE_string('dataset', './data',
                            """Path to data.""")
tf.app.flags.DEFINE_string('data_list', './data/list.pkl',
                            """Cached list of data.""")
tf.app.flags.DEFINE_string('mean_path', './data/mean.png',
                            """Path to mean of data.""")
tf.app.flags.DEFINE_string('sample_path', './samples',
                            """Path to save samples in.""")

# Training
tf.app.flags.DEFINE_integer('batch_size', 20,
                            """Number of videos to process in a batch.""")
tf.app.flags.DEFINE_integer('sample_size', 20,
                            """Number of videos to sample and save in a batch.""")
tf.app.flags.DEFINE_integer('train_epochs', 10**10,
                            """Number of training epochs.""")
tf.app.flags.DEFINE_float('lrate_d', 1e-5,
                            """Learning rate for discriminator.""")
tf.app.flags.DEFINE_float('lrate_g', 1e-4,
                            """Learning rate for generator.""")
tf.app.flags.DEFINE_float('beta1_d', 0.5,
                            """beta1 for discriminator.""")
tf.app.flags.DEFINE_float('beta1_g', 0.5,
                            """beta1 for generator.""")

# Architecture
tf.app.flags.DEFINE_integer('z_dim', 100,
                            """Dimension of initial noise vector.""")
tf.app.flags.DEFINE_integer('gf_dim', 64,
                            """Conv kernel size of G.""")
tf.app.flags.DEFINE_integer('df_dim', 64,
                            """Conv kernel size of D.""")
tf.app.flags.DEFINE_integer('c_dim', 3,
                            """Number of input channels.""")
tf.app.flags.DEFINE_float('mask_penalty', 0.1,
                            """Lambda for L1 regularizer of mask.""")

# Model saving
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoints',
                            """Path to checkpoint models.""")
tf.app.flags.DEFINE_integer('checkpoint_time', 30*60,
                            """Time to save checkpoints in.""")
tf.app.flags.DEFINE_integer('sampler_time', 30*60,
                            """Time to save samples in.""")
tf.app.flags.DEFINE_integer('print_time', 60,
			    			"""Time to print loss.""")
tf.app.flags.DEFINE_boolean('calc_mean', False,
                            """Whether or not to calculate mean.""")

def main(_):


	data = Dataset(FLAGS)
	model = videoGan(flags=FLAGS,
					 batch_size=FLAGS.batch_size, 
					 sample_size = FLAGS.sample_size,
					 z_dim=FLAGS.z_dim,
					 gf_dim=FLAGS.gf_dim,
					 df_dim=FLAGS.df_dim,
					 c_dim=FLAGS.c_dim,
					 mask_penalty=FLAGS.mask_penalty)
	model.train(data)

if __name__ == '__main__':
 	tf.app.run()
