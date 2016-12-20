import sys
import tensorflow as tf
import utils
import nnet


def main(level, files_pattern):
  
  vals = utils.prepared_reader(level, files_pattern)
  model = nnet.videoGan()
  model.explore_data(vals)
    

if __name__ == '__main__':
  '''
  level: 'frame' or 'video'
  files_pattern: "train*.tfrecord"
  '''
  level, files_pattern = sys.argv[1: ]
  main(level, files_pattern)