from argparse import ArgumentParser
from scipy import sparse
import pickle


def options():
  parser = ArgumentParser()
  parser.add_argument('wtcf', type=str)
  parser.add_argument('topics', type=int)
  parser.add_argument('outpkl', type=str)
  return parser.parse_args()

def main():
  ops = options()
  wtcf = sparse.lil_matrix( (400000, ops.topics))
  vocab = []
  with open(ops.wtcf) as file:
    for i,line in enumerate(file):
      print('%d\r' % i, end='')
      info = line.strip().split()
      vocab.append(info[1])
      for j,v in map(lambda x: x.split(':'), info[2:]):
        wtcf[i,int(j)] = v
  cut = wtcf.tocsr()[:len(vocab),:]
  with open('%s' % ops.outpkl, 'wb') as file:
    pickle.dump( {'vocab': vocab, 'wtcf': cut}, file)



if __name__=='__main__':
  main()
