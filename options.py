import argparse

def get_options():
	"""
	Use this in the terminal like python3 main.py --density 3.0
	"""
	parser = argparse.ArgumentParser(prog='Simulation of multiphase flow')
	parser.add_argument("--density", help='Density of the fluid', type=float)
	opts = parser.parse_args()
	return opts