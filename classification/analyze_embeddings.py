from frankenstein import MyEmbed, load_embeddings
import numpy as np
import codecs
import gzip
from matplotlib import pyplot as plt
import json
import random
import pickle
from argparse import ArgumentParser

def options():
  parser = ArgumentParser()
  parser.add_argument('source', help='source json')
  parser.add_argument('wordlist', help='txt of words')
  parser.add_argument('classifier', help='gz of classifier')
  parser.add_argument('embeddings', help='pkl or gz of embeddings')
  parser.add_argument('output', help='what to title the imgs')
  parser.add_argument('--cosine', help='use cosine metric', action='store_true', default=False)
  
  return parser.parse_args()


def center(wemb):
  embedding = np.vstack(wemb.vocab.values())
  words = wemb.vocab.keys()
  centered = embedding - np.dot(np.ones((len(words),1)), np.dot(np.ones((1,len(words))), embedding)) / len(words)
  return MyEmbed({word: centered[i] for i,word in enumerate(words)})
  
def correlation(sourcejson, translationstxt, classifiergz, wemb, output, cosine=False):
  with codecs.open(sourcejson, 'r', 'utf-8') as file:
    source = json.load(file)
  with codecs.open(translationstxt,'r','utf-8') as file:
    target  = []
    for line, (w,coef) in zip(file, source):
      target.append( (line.strip(), coef,w))
    
  
    
  with gzip.open(classifiergz) as ifd:
    classifier, dv, label_lookup, tokenizer = pickle.load(ifd)
  model = center(load_embeddings(wemb))
  
  
  anchors = []
  all_words = []
  
  for w,coef in source:
    if w in dv.vocabulary_ and w in model:
      vec = model[w]
      if cosine:
        vec = vec/np.sqrt(np.square(vec).sum())
      all_words.append( (coef, vec))
      if abs(coef) > 1:
        anchors.append((coef, vec))
  
  if cosine:
    dist = cos_diff
  else:
    dist = euc_diff
  
  corr_array = np.zeros( (2500*len(anchors),3))
  
  for i,(coef1, vec1) in enumerate(anchors):
    for j,(coef2, vec2) in enumerate(random.sample(all_words, 2500)):
      index = i*2500 + j
      corr_array[index,0] = abs(coef1-coef2)
      corr_array[index,1] = dist(vec1, vec2)
      corr_array[index,2] = coef1
  
  corr_array = corr_array[corr_array[:,1].argsort()].T
  
  # First, the overall
  discrete = 1*(corr_array[0] > 2)
  condprob = np.cumsum(discrete) / (np.array(range(discrete.size))+1)
  cumul = np.cumsum(corr_array[0]) / (np.array(range(discrete.size))+1)
  plt.scatter(corr_array[1], condprob)
  plt.plot(corr_array[1,[0,-1]], [condprob[-1]]*2, color='red')
  plt.title('P(error|distance<d) for %s' % ('cosine' if cosine else 'euclid'))
  plt.xlabel('Distance')
  plt.ylabel('Prob')
  plt.savefig('%s_pcosine.png' % output)
  plt.show()
  

  plt.scatter(corr_array[1], cumul)
  plt.plot(corr_array[1,[0,-1]], [cumul[-1]]*2, color='red')
  plt.title('Average error for close points for %s' % ('cosine' if cosine else 'euclid'))
  plt.xlabel('Distance')
  plt.ylabel('Avg error')
  plt.savefig('%s_avgcosine.png' % output)
  plt.show()
  
  # Now just for contro
  contro = [i for i in range(corr_array.shape[1]) if corr_array[2,i] > 0]
  discrete_c = 1*(corr_array[0,contro] > 2)
  condprob_c = np.cumsum(discrete_c) / (np.array(range(discrete_c.size))+1)
  cumul_c = np.cumsum(corr_array[0,contro]) / (np.array(range(discrete_c.size))+1)
  plt.scatter(corr_array[1,contro], condprob_c)
  plt.plot(corr_array[1,[contro[0], contro[-1]]], [condprob_c[-1]]*2, color='red')
  plt.title('P(error|distance<d, contro) for %s' % ('cosine' if cosine else 'euclid'))
  plt.xlabel('Distance')
  plt.ylabel('Prob')
  plt.savefig('%s_pcosine_contro.png' % output)
  plt.show()
  

  plt.scatter(corr_array[1,contro], cumul_c)
  plt.plot(corr_array[1,[contro[0], contro[-1]]], [cumul_c[-1]]*2, color='red')
  plt.title('Average error for close contro points for %s' % ('cosine' if cosine else 'euclid'))
  plt.xlabel('Distance')
  plt.ylabel('Avg error')
  plt.savefig('%s_avgcosine_contro.png' % output)
  plt.show()

  # Non-contro
  ncontro = [i for i in range(corr_array.shape[1]) if corr_array[2,i] < 0]
  discrete_n = 1*(corr_array[0,ncontro] > 2)
  condprob_n = np.cumsum(discrete_n) / (np.array(range(discrete_n.size))+1)
  cumul_n = np.cumsum(corr_array[0,ncontro]) / (np.array(range(discrete_n.size))+1)
  plt.scatter(corr_array[1,ncontro], condprob_n)
  plt.plot(corr_array[1,[ncontro[0], ncontro[-1]]], [condprob_n[-1]]*2, color='red')
  plt.title('P(error|distance<d, noncontro) for %s' % ('cosine' if cosine else 'euclid'))
  plt.xlabel('Distance')
  plt.ylabel('Prob')
  plt.savefig('%s_pcosine_noncontro.png' % output)
  plt.show()
  

  plt.scatter(corr_array[1,ncontro], cumul_n)
  plt.plot(corr_array[1,[ncontro[0], ncontro[-1]]], [cumul_n[-1]]*2, color='red')
  plt.title('Average error for close noncontro points for %s' % ('cosine' if cosine else 'euclid'))
  plt.xlabel('Distance')
  plt.ylabel('Avg error')
  plt.savefig('%s_avgcosine_noncontro.png' % output)
  plt.show()

  

def cos_diff(a,b):
  return 1-np.dot(a,b)
def euc_diff(a,b):
  return np.sqrt(np.square(a-b).sum())
  
if __name__=='__main__':
  ops = options()
  correlation(ops.source, ops.wordlist, ops.classifier, ops.embeddings, ops.output, cosine=ops.cosine)