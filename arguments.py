import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Description of your program')
    
    # Add your command line arguments here
    parser.add_argument('-f', '--file', type=str, help='Path to a file')
    parser.add_argument('-n', '--number', type=int, help='A number')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose mode')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Return the parsed arguments
    return args

# Example usage
if __name__ == '__main__':
    args = parse_arguments()
    print(args.file)
    print(args.number)
    print(args.verbose)