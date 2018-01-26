import gensim, re, codecs
from argparse import ArgumentParser
import json
import numpy as np

		
def main2():
	ops = options()
	lda = gensim.models.LdaModel.load(ops.filename)
	dk = gensim.corpora.Dictionary.load('%s.dk' % ops.filename)
	terms = [ [dk[j] for j,_ in lda.get_topic_terms(i, ops.n_terms)] for i in range(lda.num_topics)]
	with codecs.open('%s.keys' % ops.output, 'w', 'utf-8') as file:
		json.dump(terms, file)
	with codecs.open('%s.txt' % ops.output, 'w', 'utf-8') as file:
		for i, topic in enumerate(terms):
			file.write('%d %s\n' % (i, ', '.join(topic)))

def main():
	ops = options()
	lda = gensim.models.LdaModel.load(ops.filename)
	dk = gensim.corpora.Dictionary.load('%s.dk' % ops.filename)
	m = lda.get_topics()
	mags = [ 1./ (0.01 + m[:,i].sum()) for i in range(m.shape[1])]
	terms = []
	for i in range(m.shape[0]):
		for j in range(m.shape[1]):
			m[i,j] *= mags[j]
		terms.append( [dk[k] for k in sorted(range(m.shape[1]), key=lambda x: -m[i,x])[:ops.n_terms]])
	with codecs.open('%s.keys' % ops.output, 'w', 'utf-8') as file:
		json.dump(terms, file)
	with codecs.open('%s.txt' % ops.output, 'w', 'utf-8') as file:
		for i, topic in enumerate(terms):
			file.write('%d %s\n' % (i, ', '.join(topic)))


			
def options():
	parser = ArgumentParser()
	parser.add_argument('filename')
	parser.add_argument('n_terms', type=int)
	parser.add_argument('output')
	return parser.parse_args()
	
if __name__=='__main__':
	main()