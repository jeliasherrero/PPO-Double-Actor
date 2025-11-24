"""
	This file contains the arguments to parse at command line.
	File main.py will call get_args, which then the arguments
	will be returned.
"""
import argparse

def get_args():
	"""
		Description:
		Parses arguments at command line.

		Parameters:
			None

		Return:
			args - the arguments parsed
	"""
	parser = argparse.ArgumentParser()

	parser.add_argument('--mode', dest='mode', type=str, default='train')              # can be 'train' or 'test'
	parser.add_argument('--actor_model', dest='actor_model', type=str, default='')     # your actor model filename
	parser.add_argument('--critic_model', dest='critic_model', type=str, default='')   # your critic model filename
	parser.add_argument('--decision_model', dest='decision_model', type=str, default='')
	parser.add_argument('--class_dim', dest='class_dim', type=int, default=2)
	parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
	parser.add_argument('--gpu', dest='gpu', type=bool, default=False)

	args = parser.parse_args()

	return args
