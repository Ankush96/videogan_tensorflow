import tensorflow as tf
from tensorflow.python.platform import gfile
import glob
import os
import random
import sys
import numpy as np
from skimage import io
from skimage.transform import resize
from options import Options


class Dataset(object):
	def __init__(self, 
				 calc_mean=False):
		print("Loading paths of video data")

		self._calc_mean = calc_mean
		self._train = self._load_data(Options.dataset)

		# Load mean . Shape [32*128, 128, 3]
		self._mean = io.imread(Options.mean_path)
		# Reshape to [32, 128, 128, 3]
		self._mean = np.reshape(self._mean, [-1,128,128,3])
		new_mean = np.zeros(shape=Options.video_shape)
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
		pattern = "*.jpg"
		for dr,_,_ in os.walk(start_dir):
			data.extend(glob.glob(os.path.join(dr,pattern)))

		random.shuffle(data)

		if self._calc_mean:

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
						if ims.shape[0] > Options.video_shape[0]: # 32
							ims = ims[:Options.video_shape[0], :, :, :]
						elif ims.shape[0] < Options.video_shape[0]: #32
							# Copy the last frame till 32
							frames = ims.shape[0]
							repeats = [1 for i in range(frames)]
							repeats[-1] = Options.video_shape[0] - frames + 1
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

			batch = np.zeros(shape=[ind2-ind1]+Options.video_shape) # Prefixing with batch_size

			for video_num,path in enumerate(batch_paths):
				# Read image [a*128, 128, 3]
				ims = io.imread(path)
				ims = ims.reshape(-1,128,128,3)

				# We have to make sure a=32
				if ims.shape[0] > Options.video_shape[0]: # 32
					# Use only 1st 32 frames
					ims = ims[:Options.video_shape[0], :, :, :]
				elif ims.shape[0] < Options.video_shape[0]: #32
					# Copy the last frame till 32
					frames = ims.shape[0]
					repeats = [1 for i in range(frames)]
					repeats[-1] = Options.video_shape[0] - frames + 1
					ims = np.repeat(ims, repeats, axis=0)

				new_ims = np.zeros(shape=Options.video_shape)
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
	if not os.path.exists(Options.sample_path):
		os.makedirs(Options.sample_path)

	folder = 'Epoch_%s_Counter_%s'%(str(epoch),str(counter))
	folder = os.path.join(Options.sample_path,folder)
	if not os.path.exists(folder):
		os.makedirs(folder)

	# Load the mean
	_mean = io.imread(Options.mean_path)
	_mean = np.reshape(_mean, [-1,128,128,3])
	new_mean = np.zeros(shape=Options.video_shape)
	for i, im in enumerate(_mean):
		new_mean[i] = resize(im,(64,64,3),order=3)
	_mean = new_mean	
	# samples shape is [sample_size]+video_shape = [ss, 32, 64, 64, 3]
	for sample_num in range(samples.shape[0]):
		sample = samples[sample_num]
		sample = np.reshape(sample, [-1,64,3])
		name = str(sample_num)+'.jpg'
		name = os.path.join(folder,name)
		io.imsave(name, sample)
