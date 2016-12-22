import tensorflow as tf


class Options(object):

	# training
	batch_size = 64
	sample_size = 64
	train_epochs = 100000
	lrate = 1e-4
	beta1 = 0.5


	# architecture
	z_dim = 100
	gf_dim = 64
	df_dim = 64
	gfc_dim = 1024
	dfc_dim = 1024
	c_dim = 3
	mask_penalty = 0.1


	# data
	video_shape = [32,64,64,3]
	dataset = './data'
	mean_path = './data/mean.npz'


	def prefix():
		prefix = "videogan_"
		prefix += "bs_%d_" % (Options.batch_size)
		prefix += "ss_%d_" % (Options.sample_size)
		prefix += "te_%d_" % (Options.train_epochs)
		prefix += "lr_%d_" % (Options.lrate)
		return prefix

		
	# Model saving
	def checkpoint_dir():
		return "./checkpoints/" + Options.prefix()
	
    