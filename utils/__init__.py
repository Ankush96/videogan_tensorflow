import tensorflow as tf
from tensorflow.python.platform import gfile
import glob
import os


class Dataset(object):
  def __init__(self):
    print("Loading paths of video data")
    self._train = self._load_data(Options.train_dataset)
    print("Read dataset")

  def _load_data(self, data_dir):
    data = glob(os.path.join(data_dir,"*.jpg"))


  def train_iter(self):
    return self._get_iter(self._train)

  def _get_iter(self, data):
    n = len(data)
    num_batches = n // Options.batch_size

    for i in range(num_batches-1):
      # TODO: Some processing
      ind1 = i*Options.batch_size
      ind2 = min(n, (i + 1) * Options.batch_size)

      yield data[ind1:ind2]


def save_samples(samples):
  pass