import tensorflow as tf
from tensorflow.python.platform import gfile
import glob
import os
import random
import numpy as np
from skimage import io
from skimage.transform import resize
from options import Options


class Dataset(object):
	def __init__(self):
	 print("Loading paths of video data")
	 self._train = self._load_data(Options.dataset)
	 print("Read dataset")

	def _load_data(self, start_dir):
	 data = []
	 pattern = "*.jpg"
	 for dr,_,_ in os.walk(start_dir):
	   data.extend(glob.glob(os.path.join(dr,pattern)))

	 random.shuffle(data)

	 #-----------Comment out after use once----------
	 '''
		print("Checking integrity of files and calculationg mean. This will take several minutes and needs to be done only once.")
		print("Once this is done, please comment out the code in utils/__init__.py")
		# TODO: Calculate mean and save
		for a in data:
			try:
			im = io.imread(a)
			except ValueError:
			print("Delete the file(s) %s. Restart the program after commenting out these lines"%a)
	 '''
	 #-----------------------------------------------
	 return data

	def train_iter(self):
	 return self._get_iter(self._train)

	def _get_iter(self, data):
		n = len(data)
		num_batches = n // Options.batch_size

		for i in range(num_batches-1):

			ind1 = i*Options.batch_size
			ind2 = min(n, (i + 1) * Options.batch_size)

			batch_paths = data[ind1:ind2]

			batch = np.zeros(shape=[ind2-ind1]+Options.video_shape)

			for video_num,path in enumerate(batch_paths):
				#print("Processing video %i"%video_num)
				ims = io.imread(path)
				ims = ims.reshape(-1,128,128,3)

				if ims.shape[0] > Options.video_shape[0]: # 32
				  ims = ims[:Options.video_shape[0], :, :, :]

				new_ims = np.zeros(shape=Options.video_shape)
				for i, im in enumerate(ims):
				  new_ims[i] = resize(im,(64,64,3),order=3)

				batch[video_num] = new_ims

			yield batch


def save_samples(samples):
	pass