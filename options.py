import tensorflow as tf


class Options(object):
	batch_size = 32
	train_epochs = 100
	lstm_layers = 1
	keep_prob = 0.8
	lstm_dim = 128

	num_classes = 4816
	lrate = 1e-4
	beta1 = 0.5

    