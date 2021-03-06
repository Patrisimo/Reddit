from argparse import ArgumentParser
import numpy as np
import pickle
import codecs, json

def options():
  parser = ArgumentParser()
  parser.add_argument('wtcf', type=str)
  parser.add_argument('topics', type=int)
  parser.add_argument('outpkl', type=str)
  parser.add_argument('vocab', type=str)
  return parser.parse_args()

def main():
  ops = options()
  with codecs.open(ops.vocab, encoding='utf-8') as file:
    vocab = json.load(file)
  vocab = { k:i for i,k in enumerate(vocab)}
  
  wtcf = np.zeros( (len(vocab), ops.topics), dtype=np.int32)
  
  with codecs.open(ops.wtcf, 'r', 'utf-8') as file:
    for i,line in enumerate(file):
      print('%d\r' % i, end='')
      info = line.strip().split()
      if info[1] in vocab:
        line = vocab[info[1]]
      else:
        continue
      for j,v in map(lambda x: x.split(':'), info[2:]):
        wtcf[line,int(j)] = v
  cut = wtcf
  with open('%s' % ops.outpkl, 'wb') as file:
    pickle.dump( {'vocab': vocab, 'wtcf': cut}, file)



if __name__=='__main__':
  main()