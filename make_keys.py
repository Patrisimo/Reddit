import gensim, re, codecs
from argparse import ArgumentParser
import json

		
def main():
	ops = options()
	lda = gensim.models.LdaModel.load(ops.filename)
	dk = gensim.corpora.Dictionary.load('%s.dk' % ops.filename)
	terms = [ [dk[j] for j,_ in lda.get_topic_terms(i, ops.n_terms)] for i in range(lda.num_topics)]
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