from gensim.models import KeyedVectors
import numpy as np
import gzip
import pickle
import random
from scipy import stats


wemb = "spanish-all.kv"
model = 'spanish_round9_nb3.model.gz'

kv = KeyedVectors.load(wemb)
with gzip.open(model) as ifd:
  classifier, dv, label_lookup = pickle.load(ifd)

  
  
common_words = []
common_vectors = []
common_labels = []
common_counts = []

for w in kv.vocab:
  if w in dv.vocabulary_:
    i = dv.vocabulary_[w]
    yes, no = classifier.feature_count_[:,i]
    if yes + no > 20:
      common_words.append(w)
      common_vectors.append(kv[w])
      common_labels.append( yes - no)
      common_counts.append( (yes,no))

common_matrix = np.vstack(common_vectors)
positive = [0, 0, 0, 0, 0, 0]
negative = [0, 0, 0, 0, 0, 0]
for word, vector, label in zip(common_words, common_vectors, common_labels):
  
  sims = np.dot(common_matrix, vector)
  neighbs = [i for i,_ in sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[1:6]]
  scores = [ common_labels[i] * label for i in neighbs]
  if label > 0:
    positive[sum(s > 0 for s in scores)] += 1
  if label < 0:
    negative[sum(s > 0 for s in scores)] += 1    


pos_y = [0,0,sum(c[0] for c in common_counts)]
pos_n = [0,0,sum(c[1] for c in common_counts)]
neg_y = [0,0,sum(c[0] for c in common_counts)]
neg_n = [0,0,sum(c[1] for c in common_counts)]


for word, vector, label in zip(common_words, common_vectors, common_labels):
  
  sims = np.dot(common_matrix, vector)
  neighbs = [i for i,_ in sorted(enumerate(sims), key=lambda x: x[1], reverse=True)]
  ys = sum(common_counts[i][0] for i in neighbs[1:6])
  ns = sum(common_counts[i][1] for i in neighbs[1:6])
  if label > 0:
    pos_y[0] += ys
    pos_n[0] += ns
  if label < 0:
    neg_y[0] += ys
    neg_n[0] += ns
    
  ys = sum(common_counts[i][0] for i in neighbs[-5:])
  ns = sum(common_counts[i][1] for i in neighbs[-5:])
  if label > 0:
    pos_y[1] += ys
    pos_n[1] += ns
  if label < 0:
    neg_y[1] += ys
    neg_n[1] += ns
  

w_labels = []
w_embs = []
for w in kv.vocab:
  if w in dv.vocabulary_:
    i = dv.vocabulary_[w]
    yes, no = classifier.feature_count_[:,i]
    w_embs.append(kv[w])
    w_labels.append( (w, ci_lower_bound(yes, yes+no, 0.95), ci_lower_bound(yes, yes+no, 0.95)))
    
    
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
  
 wemb = KeyedVectors.load_word2vec_format('SBW-vectors-300-min5.bin.gz', binary=True)
 
from sklearn.cross_decomposition import PLSRegression 
def doPLSR(wemb, labels):
  