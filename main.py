import sys
import tensorflow as tf
import utils
import nnet
import argparse

def main():

	parser = argparse.ArgumentParser("Parses hyperparameters")
    parser.add_argument("--calc_m", type=bool, default=False)
    p = parser.parse_args(sys.argv[1:])

	data = utils.Dataset(p.calc_m)
	model = nnet.videoGan()
	model.train(data)

if __name__ == '__main__':
 	main()
