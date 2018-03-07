import numpy as np
import gensim
from argparse import ArgumentParser
import codecs
import re
import gzip
from os import path
import tarfile
import bz2
import unicodedata

def options():
  parser = ArgumentParser()
  parser.add_argument('input')
  parser.add_argument('output')
  parser.add_argument('--wiki', action='store_true', default=False)
  return parser.parse_args()
	
def main():
	ops = options()
	model = gensim.models.Word2Vec(MyCorpus(ops.input, ops.wiki), min_count=10).wv
	model.save('%s.kv' % ops.output)

  
  

# Read the data, one comment at a time
class MyCorpus:
  def __init__(self, fname, wiki=False):
    try:
      xrange
      self.python2 = True
    except NameError:
      self.python2 = False
    if path.splitext(fname)[1] == '.tgz':
      self.file = Tgzfile(fname)
    elif wiki:
      self.file = WikiReader(fname)
    else:
      reader = codecs.getreader('utf-8')
      self.file = reader(gzip.open(fname))
    self.sents = []
    self.word_regex = re.compile('\w(?:\w|[-\'])*\w', re.UNICODE)
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
          if self.python2: # accented characters will now be capitalized
            self.sents.append(re.findall(self.word_regex, unicodedata.normalize('NFD', s.lower()).encode('ascii','ignore')))
          else:
            self.sents.append(re.findall(self.word_regex, s.lower())) 
      return self.sents.pop(0)
    except StopIteration:
      self.file.close()
      raise StopIteration
    except Exception as e:
      print(e)
  def close(self):
    self.file.close()
    
class Tgzfile:
  def __init__(self, fname):
    self.closed = False
    self.file = tarfile.open(fname)
    self.members = self.file.getmembers()
    print('Opening %s' % self.members[0].name)
    self.current = self.file.extractfile(self.members[0].name)
    
  
  def readline(self):
    if self.file.closed:
      raise StopIteration
    while not self.file.closed:
      try:
        line = self.current.readline().decode('latin-1')
        return '1\t2\t%s' % line
      except StopIteration:
        self.current.close()
        self.members.pop(0)
        if len(self.members) == 0:
          print('Reading complete')
          self.close()
        else:
          print('Stopping now')
          self.close()
          print('Opening %s' % self.members[0].name)
          self.current = self.file.extractfile(self.members[0].name)
    raise StopIteration
    
  def close(self):
    self.closed = True
    self.file.close()
    self.current.close()
      

class WikiReader:
  def __init__(self, fname):
    self.closed = False
    reader = codecs.getreader('latin-1')
    self.file = reader(bz2.BZ2File(fname))
  def readline(self):
    if self.file.closed:
      raise StopIteration
    while not self.file.closed:
      try:
        doc_text = ''
        doc_started = False
        for line in self.file:
          if line.strip() == '<content>':
            doc_started = True
          elif line.strip() == '</content>':
            break
          elif doc_started:
            doc_text += re.sub('<[^>]*>', '', line)
        return '1\t2\t%s' % doc_text
      except StopIteration:
          print('Reading complete')
          self.close()
    raise StopIteration
    
  def close(self):
    self.closed = True
    self.file.close()
    
          
		



if __name__=='__main__':
	main()