import numpy as np
import gensim
from argparse import ArgumentParser
import codecs
import re
import gzip

def options():
	parser = ArgumentParser()
	parser.add_argument('input')
	parser.add_argument('output')
	return parser.parse_args()
	
def main():
	ops = options()
	model = gensim.models.Word2Vec(MyCorpus(ops.input), min_count=10).wv
	model.save('%s.kv' % ops.output)


# Read the data, one comment at a time
class MyCorpus:
  def __init__(self, fname):
    reader = codecs.getreader('utf-8')
    self.file = reader(gzip.open(fname))
    self.sents = []
  def __iter__(self):
    return self
  def __next__(self):
    if len(self.sents) > 0:
      return self.sents.pop(0)
    if self.file.closed:
      raise StopIteration
    
    try:
      while len(self.sents) == 0:
        info = []
        while len(info) < 3:
          line = self.file.readline()
          if len(line) == 0:
            raise StopIteration
          info = line.split('\t',2)
        for s in re.findall('[^.?!]+', info[2].strip()):
          self.sents.append(re.findall('\w(?:\w|[-\'])*\w', s.lower()))
      return self.sents.pop(0)
    except StopIteration:
      self.file.close()
      raise StopIteration
      
    
		



if __name__=='__main__':
	main()