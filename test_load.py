import argparse
import sys

import tensorflow as tf

def main():
	parser = argparse.ArgumentParser(description='Experiment parameters')
	parser.add_argument('--load_model', dest='load_model', default=None,
                      help='the checkpoint of the model you want to load from')
	args, unknown = parser.parse_known_args(sys.argv[1:])
	if not args.load_model:
		print('None')
	else:
		

main()

