from options import get_options

def main():
	# Include the options from the parser
	opts = get_options()

	print(opts.density) # Example

if __name__ == "__main__":
	main()
