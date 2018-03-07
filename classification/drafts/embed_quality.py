from gensim.models import KeyedVectors
import numpy as np
import gzip
import pickle
import random
from scipy import stats
from argparse import ArgumentParser
import codecs
import re

# Take in a word embedding and a list of words with scores, return 
# 1. Correlation between scores of words and scores of neighbors
# 2. idk we'll figure this out later


class MyEmbed():
  def __init__(self, embed):
    self.vocab = embed
  def __getitem__(self, item):
    return self.vocab[item]

def options():
  parser = ArgumentParser()
  
  parser.add_argument('embeddings', help='.kv or .pkl of word embeddings')
  parser.add_argument('model', help='classifier model')
  parser.add_argument('--min', help='minimum number of occurances for words to be considered, requires --corpus', default=0, type=int)
  parser.add_argument('--corpus', help='original corpus of classifier')
  
  return parser.parse_args()

     
def load_embeddings(fname):
  ext = path.splitext(fname)[1]
  
  if ext == '.kv':
    model = KeyedVectors.load(fname)
  elif ext == '.pkl':
    with open(fname, 'rb') as file:
      model = pickle.load(file, encoding='latin-1')
  return model
        
  
  
def evaluate(ops):
  with gzip.open(ops.model) as ifd:
    classifier, dv, label_lookup = pickle.load(ifd)
  names = dv.get_feature_names()
  
  embed = load_embeddings(ops.embeddings)
  
  if ops.min > 0:
    if ops.corpus:
      target_vocab = {}
      reader = codecs.getreader('utf-8')
      with reader(gzip.open(ops.corpus)) as file:
        for line in file:
          info = line.split('\t',2)
          for w in re.findall('\w(?:\w|[-\'])*\w', info[2].lower()):
            target_vocab[w] = 1+target_vocab.get(w,0)
      allowed = [ dv.vocabulary_[w] for w,c in target_vocab.items() if c > ops.min and w in names ]
      coefs = [ (i, classifier.coef_[0,i]) for i in allowed]
    else:
      logging.error('--corpus option required if --min set above zero')
  else:
    coefs = enumerate(classifier.coef_[0])
  
  tokens = [ (names[i], i, r) for i, r in coefs ]
  
  # Step 1: Neighbors
  embeddings = []
  labels = {}
  filtered_tokens = []
  for name, i, score in tokens:
    if name in embed.vocab:
      embeddings.append(embed[name])
      labels[len(labels)] = score
      filtered_tokens.append( (i, score, embed[name]))
  embeddings = np.vstack(embeddings)
  
  neighbors = []
  for i, score, vec in filtered_tokens:
    sims = sorted(enumerate(np.dot(vec, embeddings)))
    neighbors.append( sims[1:11])
  
  # Step 1a: Correlation
  A = np.zeros( (len(neighbors),))
  B = np.zeros( (len(neighbors),))
  for i, neighbs in enumerate(neighbors):
    A[i] = labels[i]
    B[i] = sum( labels[y]*d for y,d in neighbs) / sum(d for _,d in neighbs)
    
  slope, inter, rval, pval, stderr = stats.linregress(A,B)
  print('Neighbor = %.3f Anchor + %.3f' % (slope, inter))
  print('r-value: %f' % rval)
  print('p-value: %f' % pval)
  print('Standard error: %f' % stderr)


  # Step 1b: Counts
  counts = np.zeros( (2,2)) # counts[1,0] is # neighbors of non-contro word that are contro
  for i, neighbs in enumerate(neighbors):
    row = 0 if labels[i] < 0 else 1
    for j, _ in neighbs:
      col = 0 if labels[j] < 0 else 1
      counts[row,col] += 1
  
  print('%6s|# Yes|# No ')
  print('Contro|%5d|%5d' % (counts[0,0], counts[0,1]))
  print('   Not|%5d|%5d' % (counts[1,0], counts[1,1]))
  
  
  # Step 1c: Averages
  sums = [0,0]
  cts = [0,0]
  for i, neighbs in enumerate(neighbors):
    index = 0 if labels[i] < 0 else 1
    for j, _ in neighbs:
      sums[index] += labels[j]
      cts[index] += 1
      
  print('Average score of negative anchors: %f' % (sums[0] / cts[0]))
  print('Average score of postiive anchors: %f' % (sums[1] / cts[1]))
  print('Average score of all neighbors: %f' % ( (sum(sums) / sum(cts))))
   
    
def plot_with_labels(low_dim_embs, labels, filename, annotate=True):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, (label, cy, cn) in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y, color=(cy, abs(cy-cn)/(cy+cn), cn))
    if annotate:
      plt.annotate(label,
                   xy=(x, y),
                   xytext=(5, 2),
                   textcoords='offset points',
                   ha='right',
                   va='bottom')

  plt.savefig(filename)
  plt.show()
 
def ci_lower_bound(pos, n, confidence, upper=False):
  # Taken from http://www.evanmiller.org/how-not-to-sort-by-average-rating.html
  if n == 0:
    return 0
  z = -stats.norm.ppf( 1-(1-confidence)/2)
  if upper:
    z = -z
  p_hat = 1.0*pos/n
  return (p_hat + z*z/(2*n) + z * np.sqrt( (p_hat*(1-p_hat)+z*z/(4*n))/n))/(1+z*z/n)
  
 
 
if __name__=='__main__':
  evaluate(options())