import gensim, re, codecs
from argparse import ArgumentParser


class MyCorpus(object):
	def __init__(self, fname):
		self.fname = fname
		self.dictionary = gensim.corpora.Dictionary([self.tokenize(l) for l in codecs.open(fname,'r','utf-8')])
	def __iter__(self):
		with codecs.open(self.fname, 'r', 'utf-8') as file:
			for line in file:
				yield self.dictionary.doc2bow(self.tokenize(line))
				#yield self.tokenize(line)
	def tokenize(self, line):
		info = line.split('\t',1)[1]
		return re.findall('\w(?:\w|[-\'])+\w', info.lower())
		
def main():
	ops = options()
	corpus = MyCorpus(ops.filename)
	lda = gensim.models.LdaModel(corpus, num_topics=ops.topics)
	lda.save(ops.output)
	corpus.dictionary.save('%s.dk' % ops.output)
	
def options():
	parser = ArgumentParser()
	parser.add_argument('filename')
	parser.add_argument('topics')
	parser.add_argument('output')
	return parser.parse_args()
	
if __name__=='__main__':
	main()