import numpy as np
import gensim
from argparse import ArgumentParser
import codecs, re

def options():
	parser = ArgumentParser()
	parser.add_argument('input')
	parser.add_argument('output')
	return parser.parse_args()
	
def main():
	ops = options()
	model = gensim.models.Word2Vec(read_data(ops.input), min_count=10).wv
	model.save('%s.kv' % ops.output)


# Read the data, one comment at a time
def read_data(filename):
	sents = []
	with codecs.open(filename, 'r', 'utf-8') as file:
		for line in file:
			info = line.split('\t',1)
			if len(info) < 2:
				continue
			for s in re.findall('[^.?!]+', info[1]):
				sents.append(' '.join(re.findall('\w(?:\w[-\'])*\w', line.lower())))
	return sents
		



if __name__=='__main__':
	main()