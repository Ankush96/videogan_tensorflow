import tensorflow as tf
import numpy as np
import sys
from nnet import modules as md
import utils
import os
import time
import glob

global Options

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
		self.d_bn1 = md.batch_norm(name='d_bn1')
		self.d_bn2 = md.batch_norm(name='d_bn2')
		self.d_bn3 = md.batch_norm(name='d_bn3')

		# Batch norms for static branch of generator
		self.g_sbn0 = md.batch_norm(name='g_sbn0')
		self.g_sbn1 = md.batch_norm(name='g_sbn1')
		self.g_sbn2 = md.batch_norm(name='g_sbn2')
		self.g_sbn3 = md.batch_norm(name='g_sbn3')

		# Batch norms for video branch of generator
		self.g_vbn0 = md.batch_norm(name='g_vbn0')
		self.g_vbn1 = md.batch_norm(name='g_vbn1')
		self.g_vbn2 = md.batch_norm(name='g_vbn2')
		self.g_vbn3 = md.batch_norm(name='g_vbn3')

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

		self.z_sum = md.histogram_summary("z", self.z)

		# generated videos, L1 regularizer penalty from generator
		self.G, self.g_loss_penalty = self.generator(self.z)
		# sigmoid of logits, logits for real videos from discriminator
		self.D, self.D_logits = self.discriminator(self.videos)

		# sample videos from sampler
		self.sampler = self.sampler(self.z)
		# sigmoid of logits, logits for fake videos from discriminator
		self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
		
		self.d_sum = md.histogram_summary("d", self.D)
		self.d__sum = md.histogram_summary("d_", self.D_)

		# For real videos, logits must be matched against 1s
		self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
		# For fake videos, logits must be matched against 0s
		self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
		# Generator loss is modelled
		self.g_loss_no_penalty = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

		self.d_loss_real_sum = md.scalar_summary("d_loss_real", self.d_loss_real)
		self.d_loss_fake_sum = md.scalar_summary("d_loss_fake", self.d_loss_fake)
		
		# Total discriminator loss
		self.d_loss = self.d_loss_real + self.d_loss_fake

		self.g_loss_no_penalty_sum = md.scalar_summary("g_loss_no_penalty", self.g_loss_no_penalty)
		self.g_loss_penalty_sum = md.scalar_summary("g_loss_penalty", self.g_loss_penalty)

		# Total generator loss
		self.g_loss = self.mask_penalty*self.g_loss_penalty + self.g_loss_no_penalty

		self.g_loss_sum = md.scalar_summary("g_loss", self.g_loss)
		self.d_loss_sum = md.scalar_summary("d_loss", self.d_loss)

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
		
		h0 = md.lrelu(md.conv3d(video, self.df_dim, name='d_h0_conv'))
		# print("video -> h0 - shape = ", h0.get_shape().as_list())		# [bs, 16, 32, 32, 64]
		
		h1 = md.lrelu(self.d_bn1(md.conv3d(h0, self.df_dim*2, name='d_h1_conv')))
		# print("h0 -> h1 - shape = ", h1.get_shape().as_list())		# [bs, 8, 16, 16, 128]
		
		h2 = md.lrelu(self.d_bn2(md.conv3d(h1, self.df_dim*4, name='d_h2_conv')))
		# print("h1 -> h2 - shape = ", h2.get_shape().as_list())		# [bs, 4, 8, 8, 256]
		
		h3 = md.lrelu(self.d_bn3(md.conv3d(h2, self.df_dim*8, name='d_h3_conv')))
		# print("h2 -> h3 - shape = ", h3.get_shape().as_list())		# [bs, 2, 4, 4, 512]
		
		h4 = md.linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
		# print("h3 -> h4 - shape = ", h4.get_shape().as_list())		# [bs, 1]

		return tf.nn.sigmoid(h4), h4


	def generator(self, z, y=None):
		s = 64
		s2, s4, s8, s16, s32 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32)
		
		# print("Inside generator")
		# print("z- shape = ", z.get_shape().as_list())				   # [None, 100]
		
		# s stands for static part
		self.sz_, self.sh0_w, self.sh0_b = md.linear(z, self.gf_dim*8*s16*s16, 'g_sh0_lin', with_w=True)
		# print("z -> sz_ - shape = ", self.sz_.get_shape().as_list())	# [None, 8192]
		
		self.sh0 = tf.reshape(self.sz_, [-1, s16, s16, self.gf_dim * 8])
		sh0 = tf.nn.relu(self.g_sbn0(self.sh0))
		# print("sz_ -> sh0 - shape = ", sh0.get_shape().as_list())		# [None, 4, 4, 512]
		
		self.sh1, self.sh1_w, self.sh1_b = md.deconv2d(sh0, 
			[self.batch_size, s8, s8, self.gf_dim*4], name='g_sh1', with_w=True)
		sh1 = tf.nn.relu(self.g_sbn1(self.sh1))
		# print("sh0 -> sh1 - shape = ", sh1.get_shape().as_list())		# [bs, 8, 8, 256]

		sh2, self.sh2_w, self.sh2_b = md.deconv2d(sh1,
			[self.batch_size, s4, s4, self.gf_dim*2], name='g_sh2', with_w=True)
		sh2 = tf.nn.relu(self.g_sbn2(sh2))
		# print("sh1 -> sh2 - shape = ", sh2.get_shape().as_list())		# [bs, 16, 16, 128]

		sh3, self.sh3_w, self.sh3_b = md.deconv2d(sh2,
			[self.batch_size, s2, s2, self.gf_dim*1], name='g_sh3', with_w=True)
		sh3 = tf.nn.relu(self.g_sbn3(sh3))
		# print("sh2 -> sh3 - shape = ", sh3.get_shape().as_list())		# [bs, 32, 32, 64]

		sh4, self.sh4_w, self.sh4_b = md.deconv2d(sh3,
			[self.batch_size, s, s, self.c_dim], name='g_sh4', with_w=True)
		# print("sh3 -> sh4 - shape = ", sh4.get_shape().as_list())		# [bs, 64, 64, 3]

		background = tf.nn.tanh(sh4)

		# Extending static part over time by replicating. Reshape is needed to increase dimension
		background = tf.tile(tf.reshape(background,[self.batch_size, 1, s, s, self.c_dim]), [1, s2, 1, 1, 1])
		# print("sh4-> background -> tile(background) - shape = ", background.get_shape().as_list())	# [bs, 32, 64, 64, 3]

		# v stands for video part
		self.vz_, self.vh0_w, self.vh0_b = md.linear(z, self.gf_dim*8*s32*s16*s16, 'g_vh0_lin', with_w=True)
		# print("z -> vz_ - shape = ", sh1.get_shape().as_list())			# [bs, 8, 8, 256]

		self.vh0 = tf.reshape(self.vz_, [-1, s32, s16, s16, self.gf_dim * 8])
		vh0 = tf.nn.relu(self.g_vbn0(self.vh0))
		# print("vz_ -> vh0 - shape = ", vh0.get_shape().as_list())		# [None, 2, 4, 4, 512]

		self.vh1, self.vh1_w, self.vh1_b = md.deconv3d(vh0, 
			[self.batch_size, s16, s8, s8, self.gf_dim*4], name='g_vh1', with_w=True)
		vh1 = tf.nn.relu(self.g_vbn1(self.vh1))
		# print("vh0 -> vh1 - shape = ", vh1.get_shape().as_list())		# [bs, 4, 8, 8, 256]

		vh2, self.vh2_w, self.vh2_b = md.deconv3d(vh1,
			[self.batch_size, s8, s4, s4, self.gf_dim*2], name='g_vh2', with_w=True)
		vh2 = tf.nn.relu(self.g_vbn2(vh2))
		# print("vh1 -> vh2 - shape = ", vh2.get_shape().as_list())		# [bs, 8, 16, 16, 128]

		vh3, self.vh3_w, self.vh3_b = md.deconv3d(vh2,
			[self.batch_size, s4, s2, s2, self.gf_dim*1], name='g_vh3', with_w=True)
		vh3 = tf.nn.relu(self.g_vbn3(vh3))
		# print("vh2 -> vh3 - shape = ", vh3.get_shape().as_list())		# [bs, 16, 32, 32, 64]

		mask_out, mask_out_w, mask_out_b = md.deconv3d(vh3,
			[self.batch_size, s2, s, s, 1], name='g_mask', with_w=True)

		mask_out = tf.nn.sigmoid(mask_out)
		# print("vh3 -> mask_out - shape = ", mask_out.get_shape().as_list())	# [bs, 32, 64, 64, 1]

		vh4, self.vh4_w, self.vh4_b = md.deconv3d(vh3,
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
		self.sz_ = md.linear(z, self.gf_dim*8*s16*s16, 'g_sh0_lin')

		self.sh0 = tf.reshape(self.sz_, [-1, s16, s16, self.gf_dim * 8])
		sh0 = tf.nn.relu(self.g_sbn0(self.sh0))

		self.sh1 = md.deconv2d(sh0, 
			[self.batch_size, s8, s8, self.gf_dim*4], name='g_sh1')
		sh1 = tf.nn.relu(self.g_sbn1(self.sh1))

		sh2 = md.deconv2d(sh1,
			[self.batch_size, s4, s4, self.gf_dim*2], name='g_sh2')
		sh2 = tf.nn.relu(self.g_sbn2(sh2))

		sh3 = md.deconv2d(sh2,
			[self.batch_size, s2, s2, self.gf_dim*1], name='g_sh3')
		sh3 = tf.nn.relu(self.g_sbn3(sh3))

		sh4 = md.deconv2d(sh3,
			[self.batch_size, s, s, self.c_dim], name='g_sh4')

		background = tf.nn.tanh(sh4)

		# Extending static part over time by replicating. Reshape is needed to increase dimension
		background = tf.tile(tf.reshape(background,[self.sample_size, 1, s, s, self.c_dim]), [1, s2, 1, 1, 1])

		# v stands for video part
		self.vz_ = md.linear(z, self.gf_dim*8*s32*s16*s16, 'g_vh0_lin')

		self.vh0 = tf.reshape(self.vz_, [-1, s32, s16, s16, self.gf_dim * 8])
		vh0 = tf.nn.relu(self.g_vbn0(self.vh0))

		self.vh1 = md.deconv3d(vh0, 
			[self.sample_size, s16, s8, s8, self.gf_dim*4], name='g_vh1')
		vh1 = tf.nn.relu(self.g_vbn1(self.vh1))

		vh2 = md.deconv3d(vh1,
			[self.sample_size, s8, s4, s4, self.gf_dim*2], name='g_vh2')
		vh2 = tf.nn.relu(self.g_vbn2(vh2))

		vh3 = md.deconv3d(vh2,
			[self.sample_size, s4, s2, s2, self.gf_dim*1], name='g_vh3')
		vh3 = tf.nn.relu(self.g_vbn3(vh3))

		mask_out = md.deconv3d(vh3,
			[self.sample_size, s2, s, s, 1], name='g_mask')

		mask_out = tf.nn.sigmoid(mask_out)

		vh4 = md.deconv3d(vh3,
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
			self.g_sum = md.merge_summary([self.z_sum, self.d__sum,
										 self.d_loss_fake_sum, self.g_loss_sum])
			self.d_sum = md.merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
			self.writer = md.SummaryWriter("./logs", self.sess.graph)
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
						utils.save_samples(samples, epoch, counter)
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
