from argparse import ArgumentParser
import numpy as np
import pickle
import codecs, json
from scipy import sparse


def options():
  parser = ArgumentParser()
  parser.add_argument('wtcf', type=str)
  parser.add_argument('topics', type=int)
  parser.add_argument('outpkl', type=str)
  return parser.parse_args()

def main():
  ops = options()
  wtcf = sparse.lil_matrix( (100000, ops.topics), dtype=np.int32)
  vocab = {}
  with codecs.open(ops.wtcf, 'r', 'utf-8') as file:
    for i,line in enumerate(file):
      print('%d\r' % i, end='')
      info = line.strip().split()
      vocab[info[1]] = i
      if i >= wtcf.shape[0]:
        wtcf = extend(wtcf)
      for j,v in map(lambda x: x.split(':'), info[2:]):
        wtcf[i,int(j)] = v
  cut = wtcf[:len(vocab)]
  with open('%s' % ops.outpkl, 'wb') as file:
    pickle.dump( {'vocab': vocab, 'wtcf': cut.tocsr()}, file)

"""Takes an nxm scipy lil_matrix and returns a 2nxm one"""
def extend(A):
  B = sparse.lil_matrix((2*A.shape[0], A.shape[1]))
  B[:A.shape[0]] = A
  return B
if __name__=='__main__':
  main()
  
