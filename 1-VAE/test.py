import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--target', help='What')
args = parser.parse_args()
print(args.target)
