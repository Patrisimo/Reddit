from sklearn.cross_decomposition import PLSRegression 
from gensim.models import KeyedVectors
import numpy as np
import gzip
import pickle
import random
from scipy import stats
from argparse import ArgumentParser
import logging
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def options():
  parser = ArgumentParser()
  parser.add_argument('embeddings')
  parser.add_argument('model')
  parser.add_argument('output')
  parser.add_argument('--labels', action='store_true')
  return parser.parse_args()
  
  
def main(ops):
  kv = KeyedVectors.load_word2vec_format(ops.embeddings, binary=True)
  with gzip.open(ops.model) as ifd:
    classifier, dv, label_lookup = pickle.load(ifd)
  
  w_labels = []
  w_embs = []
  for w in kv.vocab:
    if w in dv.vocabulary_:
      i = dv.vocabulary_[w]
      yes, no = classifier.feature_count_[:,i]
      w_embs.append(kv[w])
      w_labels.append( (w, ci_lower_bound(yes, yes+no, 0.95), ci_lower_bound(no, yes+no, 0.95)))

  plsr = PLSRegression()
  proj_embs, scores = plsr.fit_transform(np.vstack(w_embs), np.array([l[1] for l in w_labels]))
  plot_with_labels(proj_embs, w_labels, ops.output, annotate=ops.labels)
  
  
def ci_lower_bound(pos, n, confidence, upper=False):
  # Taken from http://www.evanmiller.org/how-not-to-sort-by-average-rating.html
  if n == 0:
    return 0
  z = -stats.norm.ppf( 1-(1-confidence)/2)
  if upper:
    z = -z
  p_hat = 1.0*pos/n
  return min(1,max(0,(p_hat + z*z/(2*n) + z * np.sqrt( (p_hat*(1-p_hat)+z*z/(4*n))/n))/(1+z*z/n)))

def plot_with_labels(low_dim_embs, labels, filename, annotate=True):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, (label, cy, cn) in enumerate(labels):
    x, y = low_dim_embs[i, :]
    s = cy + cn
    if s == 0:
      s = 1
    plt.scatter(x, y, color=(cy, abs(cy-cn)/(s), cn))
    if annotate:
      plt.annotate(label,
                   xy=(x, y),
                   xytext=(5, 2),
                   textcoords='offset points',
                   ha='right',
                   va='bottom')

  plt.savefig(filename)
  plt.show()  
  
if __name__=='__main__':
  logging.basicConfig(level=logging.INFO)
  main(options())
  