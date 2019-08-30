#!/usr/bin/env python3
import pickle
import argparse
import matplotlib.pyplot as plt

# This script is for showing the saved ttr distribution graph. Because sometimes matplotlib can't show it online.
if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--file",type=str)

	args = parser.parse_args()

	fig = pickle.load(open(args.file,'rb'))
	fig.show()
	input()

