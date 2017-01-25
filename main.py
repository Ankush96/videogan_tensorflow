import sys
import tensorflow as tf
import utils
import nnet


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


	data = utils.Dataset(FLAGS)
	model = nnet.videoGan(flags=FLAGS,
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
