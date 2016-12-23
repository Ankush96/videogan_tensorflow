import sys
import tensorflow as tf
import utils
import nnet


def main():

  data = utils.Dataset()
  model = nnet.videoGan()
  model.train(data)
    

if __name__ == '__main__':
  main()
