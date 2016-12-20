import tensorflow as tf
import sys
from options import Options
from nnet import modules as md
import utils
import os


class videoGan():

	def __init__(self, 
				 video_shape=Options.video_shape, 
				 batch_size=Options.batch_size, 
				 sample_size = Options.sample_size,
				 z_dim=Options.z_dim,
				 gf_dim=Options.gf_dim,
				 df_dim=Options.df_dim,
				 gfc_dim=Options.gfc_dim,
				 dfc_dim=Options.dfc_dim,
				 c_dim=Options.c_dim,
				 mask_penalty=Options.mask_penalty):

		"""
		Args:
			batch_size: The size of batch. Should be specified before training.
			video_shape: (optional) Shape of videos. [32,64,64,3]
			z_dim: (optional) Dimension of dim for Z. [100]
			gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
			df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
			gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
			dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
			c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
			mask_penalty: (optional) Lambda for L1 regularizer of mask
		"""

		self.batch_size = batch_size
		self.video_shape = video_shape
		self.sample_size = sample_size
		self.output_size = output_size

		self.z_dim = z_dim

		self.gf_dim = gf_dim
		self.df_dim = df_dim

		self.gfc_dim = gfc_dim
		self.dfc_dim = dfc_dim

		self.c_dim = c_dim
		self.mask_penalty = mask_penalty

		# batch normalization : deals with poor initialization helps gradient flow
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


		self.videos = tf.placeholder(
			dtype = tf.float32, 
			shape = [self.batch_size] + self.video_shape,
			name = 'real_videos')
		self.sample_videos= tf.placeholder(
			dtype = tf.float32, 
			shape = [self.sample_size] + self.video_shape,
			name = 'sample_videos')
		self.z = tf.placeholder(
			dtype = tf.float32, 
			shape = [None, self.z_dim],
			name = 'z')

		self.z_sum = tf.histogram_summary("z", self.z)


		self.G, self.g_loss_penalty = self.generator(self.z)
		self.D, self.D_logits = self.discriminator(self.videos)

		self.sampler = self.sampler(self.z)
		self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
		
		self.d_sum = tf.histogram_summary("d", self.D)
		self.d__sum = tf.histogram_summary("d_", self.D_)
		self.G_sum = tf.image_summary("G", self.G)

		self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
		self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
		self.g_loss_no_penalty = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

		self.d_loss_real_sum = tf.scalar_summary("d_loss_real", self.d_loss_real)
		self.d_loss_fake_sum = tf.scalar_summary("d_loss_fake", self.d_loss_fake)
													
		self.d_loss = self.d_loss_real + self.d_loss_fake

		self.g_loss_no_penalty_sum = tf.scalar_summary("g_loss_no_penalty", self.g_loss_no_penalty)
		self.g_loss_penalty_sum = tf.scalar_summary("g_loss_penalty", self.g_loss_penalty)

		self.g_loss = self.mask_penalty*self.g_loss_penalty + self.g_loss_no_penalty

		self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)
		self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)

		t_vars = tf.trainable_variables()

		self.d_vars = [var for var in t_vars if 'd_' in var.name]
		self.g_vars = [var for var in t_vars if 'g_' in var.name]

		self.saver = tf.train.Saver()

		print("Model defined")

		for var in tf.trainable_variables():
			print(var.name, var.get_shape())


	def discriminator(self, video, y=None, reuse=False):
		if reuse:
			tf.get_variable_scope().reuse_variables()

		h0 = md.lrelu(md.conv3d(video, self.df_dim, name='d_h0_conv'))
		h1 = md.lrelu(self.d_bn1(md.conv3d(h0, self.df_dim*2, name='d_h1_conv')))
		h2 = md.lrelu(self.d_bn2(md.conv3d(h1, self.df_dim*4, name='d_h2_conv')))
		h3 = md.lrelu(self.d_bn3(md.conv3d(h2, self.df_dim*8, name='d_h3_conv')))
		h4 = md.linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

		return tf.nn.sigmoid(h4), h4


	def generator(self, z, y=None):
		s = self.output_size
		s2, s4, s8, s16, s32 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32)

		# s stands for static part
		self.sz_, self.sh0_w, self.sh0_b = md.linear(z, self.gf_dim*8*s16*s16, 'g_sh0_lin', with_w=True)

		self.sh0 = tf.reshape(self.sz_, [-1, s16, s16, self.gf_dim * 8])
		sh0 = tf.nn.relu(self.g_sbn0(self.sh0))

		self.sh1, self.sh1_w, self.sh1_b = md.deconv2d(sh0, 
			[self.batch_size, s8, s8, self.gf_dim*4], name='g_sh1', with_w=True)
		sh1 = tf.nn.relu(self.g_sbn1(self.sh1))

		sh2, self.sh2_w, self.sh2_b = md.deconv2d(sh1,
			[self.batch_size, s4, s4, self.gf_dim*2], name='g_sh2', with_w=True)
		sh2 = tf.nn.relu(self.g_sbn2(sh2))

		sh3, self.sh3_w, self.sh3_b = md.deconv2d(sh2,
			[self.batch_size, s2, s2, self.gf_dim*1], name='g_sh3', with_w=True)
		sh3 = tf.nn.relu(self.g_sbn3(sh3))

		sh4, self.sh4_w, self.sh4_b = md.deconv2d(sh3,
			[self.batch_size, s, s, self.c_dim], name='g_sh4', with_w=True)

		background = tf.nn.tanh(sh4)

		# Extending static part over time by replicating. Reshape is needed to increase dimension
		background = tf.tile(tf.reshape(background,[self.batch_size, 1, s, s, self.c_dim]), [1, s2, 1, 1, 1])

		# v stands for video part
		self.vz_, self.vh0_w, self.vh0_b = md.linear(z, self.gf_dim*8*s32*s16*s16, 'g_vh0_lin', with_w=True)

		self.vh0 = tf.reshape(self.vz_, [-1, s32, s16, s16, self.gf_dim * 8])
		vh0 = tf.nn.relu(self.g_vbn0(self.vh0))

		self.vh1, self.vh1_w, self.vh1_b = md.deconv3d(vh0, 
			[self.batch_size, s16, s8, s8, self.gf_dim*4], name='g_vh1', with_w=True)
		vh1 = tf.nn.relu(self.g_vbn1(self.vh1))

		vh2, self.vh2_w, self.vh2_b = md.deconv3d(vh1,
			[self.batch_size, s8, s4, s4, self.gf_dim*2], name='g_vh2', with_w=True)
		vh2 = tf.nn.relu(self.g_vbn2(vh2))

		vh3, self.vh3_w, self.vh3_b = md.deconv3d(vh2,
			[self.batch_size, s4, s2, s2, self.gf_dim*1], name='g_vh3', with_w=True)
		vh3 = tf.nn.relu(self.g_vbn3(vh3))

		mask_out, mask_out_w, mask_out_b = md.deconv3d(vh3,
			[self.batch_size, s2, s, s, 1], name='g_mask', with_w=True)

		mask_out = tf.nn.sigmoid(mask_out)

		vh4, self.vh4_w, self.vh4_b = md.deconv3d(vh3,
			[self.batch_size, s2, s, s, self.c_dim], name='g_vh4', with_w=True)

		foreground = tf.nn.tanh(vh4)

		# f = f*m
		foreground = tf.mul(foreground, mask_out)
		# b = b*(1-m)
		background = tf.mul(background, tf.sub(tf.constant([1.0]), mask_out))

		gen_video = tf.add(foreground, background)
		return gen_video, tf.reduce_mean(tf.reduce_sum(tf.abs(mask_out_w)))


	def sampler(self, z, y=None):
		tf.get_variable_scope().reuse_variables()

		s = self.output_size
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
		background = tf.tile(tf.reshape(background,[self.batch_size, 1, s, s, self.c_dim]), [1, s2, 1, 1, 1])

		# v stands for video part
		self.vz_ = md.linear(z, self.gf_dim*8*s32*s16*s16, 'g_vh0_lin')

		self.vh0 = tf.reshape(self.vz_, [-1, s32, s16, s16, self.gf_dim * 8])
		vh0 = tf.nn.relu(self.g_vbn0(self.vh0))

		self.vh1 = md.deconv3d(vh0, 
			[self.batch_size, s16, s8, s8, self.gf_dim*4], name='g_vh1')
		vh1 = tf.nn.relu(self.g_vbn1(self.vh1))

		vh2 = md.deconv3d(vh1,
			[self.batch_size, s8, s4, s4, self.gf_dim*2], name='g_vh2')
		vh2 = tf.nn.relu(self.g_vbn2(vh2))

		vh3 = md.deconv3d(vh2,
			[self.batch_size, s4, s2, s2, self.gf_dim*1], name='g_vh3')
		vh3 = tf.nn.relu(self.g_vbn3(vh3))

		mask_out = md.deconv3d(vh3,
			[self.batch_size, s2, s, s, 1], name='g_mask')

		mask_out = tf.nn.sigmoid(mask_out)

		vh4 = md.deconv3d(vh3,
			[self.batch_size, s2, s, s, self.c_dim], name='g_vh4')

		foreground = tf.nn.tanh(vh4)

		# f = f*m
		foreground = tf.mul(foreground, mask_out)
		# b = b*(1-m)
		background = tf.mul(background, tf.sub(tf.constant([1.0]), mask_out))

		gen_video = tf.add(foreground, background)
		return gen_video
	 

	def train(self, dataset):

		'''Args
			dataset: An object of the class dataset which has the next_batch() function
		'''
		
		d_optim = tf.train.AdamOptimizer(Options.lrate, beta1=Options.beta1) \
						  .minimize(self.d_loss, var_list=self.d_vars)
		g_optim = tf.train.AdamOptimizer(Options.learning_rate, beta1=Options.beta1) \
						  .minimize(self.g_loss, var_list=self.g_vars)
		with tf.Session() as sess:
			tf.initialize_all_variables().run()

			self.g_sum = merge_summary([self.z_sum, self.d__sum,
            self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        	self.d_sum = merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        	self.writer = SummaryWriter("./logs", self.sess.graph)

        	sample_z = np.random.uniform(-1, 1, size=(self.sample_size , self.z_dim))

        	counter = 1

        	for epoch in Options.train_epochs:
        		for sub_data in dataset.train_iter():

        			batch_z = np.random.uniform(-1, 1, [Options.batch_size, self.z_dim]).astype(np.float32)

        			# Update D network
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                        feed_dict={ self.videos: sub_data, self.z: batch_z })
                    self.writer.add_summary(summary_str, counter)

                    # Update G n
                    etwork
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.z: batch_z })
                    self.writer.add_summary(summary_str, counter)

                    
                    errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                    errD_real = self.d_loss_real.eval({self.videos: sub_data})
                    errG = self.g_loss.eval({self.z: batch_z})

                    counter += 1

                    print("Epoch: [%d], d_loss_fake: %.8f, d_loss_real: %.8f, g_loss: %.8f" \
                    % (epoch, errD_fake, errD_real, errG))

                    if np.mod(counter, 100) == 1:
                    	samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={self.z: sample_z, self.videos: sub_data}
                        )
                        utils.save_samples(samples)
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                    if np.mod(counter, 500) == 2:
                    	if not os.path.exists("./checkpoints/"):
    						os.makedirs("./checkpoints/")
                    	self.save(Options.checkpoint_dir, counter)


    def save(self, _dir, counter, sess):
    	if not os.path.exists(_dir):
    		os.makedirs(_dir)
    	self.saver.save(sess, _dir, global_step=counter)



