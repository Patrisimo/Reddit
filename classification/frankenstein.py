from argparse import ArgumentParser
import gzip
from sklearn import naive_bayes, linear_model, svm, metrics, ensemble, dummy
from sklearn.feature_extraction import DictVectorizer
import pickle
import codecs
import logging
import json
import re
from gensim.models import KeyedVectors
import numpy as np
from os import path

class MyEmbed():
  def __init__(self, embed):
    self.vocab = embed
  def __getitem__(self, item):
    return self.vocab[item]
  def __contains__(self, item):
    return item in self.vocab

def options():
  parser = ArgumentParser()
  parser.add_argument('type', help='type of classification algorithm', choices=['naive_bayes','logistic_regression', 'svm'])
  subparsers = parser.add_subparsers(help='step to run', dest='method')
  
  parser_ex = subparsers.add_parser('extract_words', help='Extract words to translate')
  parser_ex.add_argument('model', help='Filename of model to extract words from')
  parser_ex.add_argument('k', type=int, help='How many words to extract')
  parser_ex.add_argument('output', help='Name of output files, will receive extension')
  parser_ex.add_argument('--min', type=int, help='Minimum occurance cutoff for words', default=0)
  parser_ex.add_argument('--corpus', help='If using cutoff, need corpus to count words')
  
  parser_bnb = subparsers.add_parser('make', help='Create a new parser')
  
  parser_bnb.add_argument('source', help='json from extract_words')
  parser_bnb.add_argument('model', help='previous model')
  parser_bnb.add_argument('embeddings', help='.kv or .pkl of word embeddings')
  parser_bnb.add_argument('translations', help='translated from extract_words')
  parser_bnb.add_argument('corpus', help='corpus of documents')
  parser_bnb.add_argument('output', help='will receive extension')
  
  return parser.parse_args()

def extract_words_nb(ops):
  # TODO
  with gzip.open(ops.model) as ifd:
    classifier, dv, label_lookup = pickle.load(ifd)
  names = dv.get_feature_names()
  tokens = []
  
  for i in range(classifier.feature_log_prob_.shape[1]):
    ratio = classifier.feature_log_prob_[label_lookup['1'],i] \
            - classifier.feature_log_prob_[label_lookup['0'],i]
    tokens.append((i,ratio))
  
  tokens = sorted(tokens, key=lambda x: x[1], reverse=True)
  to_translate = tokens[:ops.k//2]
  to_translate.extend(tokens[-ops.k//2:])
  
  source_lang = [ (names[i], r) for i,r in to_translate]
  
  
  with codecs.open('%s.txt' % ops.output,'w','utf-8') as file:
    for w,r in source_lang:
      file.write('%s\n' % w)
  with codecs.open('%s.json' % ops.output, 'w', 'utf-8') as file:
    json.dump(source_lang, file)
  logging.info('Words saved to %s.txt, json with scores saved to %s.json' % (ops.output, ops.output))
  
      
def load_embeddings(fname):
  ext = path.splitext(fname)[1]
  
  if ext == '.kv':
    model = KeyedVectors.load(fname)
  elif ext == '.pkl':
    with open(fname, 'rb') as file:
      model = pickle.load(file, encoding='latin-1')
  return model
      
def make_nb(ops):
  with codecs.open(ops.source, 'r', 'utf-8') as file:
    source = json.load(file)
  with codecs.open(ops.translations,'r','utf-8') as file:
    target  = []
    for line, (w,r) in zip(file, source):
      target.append( (line.strip(), r,w))
    
  
    
  with gzip.open(ops.model) as ifd:
    classifier, dv, label_lookup = pickle.load(ifd)
  
  model = load_embeddings(ops.embeddings)
      
  target_wdups = []
  for w,r,s in target:
    for w_seg in w.lower().split(' '):
      target_wdups.append( (w_seg,r,s.lower()))
      
  joined_scores = {}    
  for word,r,source_word in target_wdups:
    if source_word not in dv.vocabulary_:
      continue
    counts = classifier.feature_count_[:,dv.vocabulary_[source_word]].sum()
    if r > 0:
      yes = 1./(r+1)
    else:
      yes = 1 - 1./(1-r)
    if word in joined_scores:
      old_y, old_counts = joined_scores[word]
    else:
      old_y, old_counts = 0, 0
    joined_scores[word] = ((yes*counts + old_y * old_counts) / (counts+old_counts), counts + old_counts) 


  anchors = []
  anchor_words = {}
  for w, (y, c) in joined_scores.items():
    if w in model.vocab:
      anchors.append( (y*c, model[w] / np.sqrt(np.square(model[w]).sum())))
      anchor_words[w] = y
  
  target_vocab = {}
  reader = codecs.getreader('utf-8')
  with reader(gzip.open(ops.corpus)) as file:
    for line in file:
      info = line.split('\t',2)
      for w in re.findall('\w(?:\w|[-\'])*\w', info[2].lower()):
        target_vocab[w] = 1+target_vocab.get(w,0)
        
  common_words = [(w,c) for w,c in target_vocab.items() if c > 100]
  all_words = []
  for w,c in common_words:
    if w not in model.vocab:
      continue
    if w in anchor_words:
      all_words.append( (w, anchor_words[w], c))
      continue
    vec = model[w] / np.sqrt(np.square(model[w]).sum())
    
    neighbs = [(y, np.dot(v, vec)) for y,v in anchors]
    neighbs = sorted(neighbs, key=lambda x: x[1], reverse=True)[:5]
    prob = sum( y*d for y,d in neighbs) / sum(d for _,d in neighbs)
    all_words.append( (w, prob, c))

  flp = np.zeros( (2,len(all_words)))
  feature_count = np.zeros( (2,len(all_words)))
  total_words = sum(c for _,_,c in all_words)
  y = label_lookup['1']
  n = label_lookup['0']
  all_words = sorted(all_words, key=lambda x: x[0])

  newvocabulary = {}
  newnames = []
  newmodel = classifier
  newdv = dv
  for i,(w,p,c) in enumerate(all_words):
    flp[y,i] = np.log(c) + np.log(p) - np.log(total_words) - newmodel.class_log_prior_[y]
    flp[n,i] = np.log(c) + np.log(1-p) - np.log(total_words) - newmodel.class_log_prior_[n]
    yes = int(p*c)
    no = c - yes
    feature_count[y,i] = yes
    feature_count[n,i] = no
    newnames.append(w)
    newvocabulary[w] = i
    
  newdv.feature_names_ = newnames
  newdv.vocabulary_ = newvocabulary
  newmodel.feature_log_prob_ = flp
  newmodel.feature_count_ = feature_count

  logging.info('Saving new model in %s.model.gz' % ops.output)
  with gzip.open('%s.model.gz' % ops.output,'wb') as ofd:
    pickle.dump((newmodel, newdv, label_lookup),ofd)


def extract_words_lrsvm(ops):
  # TODO
  with gzip.open(ops.model) as ifd:
    classifier, dv, label_lookup, tokenizer = pickle.load(ifd)
  names = dv.get_feature_names()
  
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
  tokens = sorted(coefs, key=lambda x: x[1])
  
  side = -1 if label_lookup['1'] == 1 else 1
  to_translate = tokens[:ops.k:side]
  #to_translate.extend(tokens[-ops.k//2:])
  
  source_lang = [ (names[i], r) for i,r in to_translate]
  
  
  with codecs.open('%s.txt' % ops.output,'w','utf-8') as file:
    for w,r in source_lang:
      file.write('%s\n' % w)
  with codecs.open('%s.json' % ops.output, 'w', 'utf-8') as file:
    json.dump(source_lang, file)
  logging.info('Words saved to %s.txt, json with scores saved to %s.json' % (ops.output, ops.output))
  
      
def make_lrsvm(ops):
  with codecs.open(ops.source, 'r', 'utf-8') as file:
    source = json.load(file)
  with codecs.open(ops.translations,'r','utf-8') as file:
    target  = []
    for line, (w,coef) in zip(file, source):
      target.append( (line.strip(), coef,w))
    
  
    
  with gzip.open(ops.model) as ifd:
    classifier, dv, label_lookup, tokenizer = pickle.load(ifd)
  model = load_embeddings(ops.embeddings)
      
  target_wdups = []
  for w,coef,s in target:
    for w_seg in w.lower().split(' '):
      target_wdups.append( (w_seg,coef,s.lower()))
      
  joined_scores = {}    
  for word,coef,source_word in target_wdups:
    if source_word not in dv.vocabulary_:
      continue
    
    if word in joined_scores:
      old_coef, old_counts = joined_scores[word]
    else:
      old_coef, old_counts = 0, 0
    joined_scores[word] = ((coef + old_coef * old_counts) / (1+old_counts), 1 + old_counts) 


  anchors = []
  anchor_words = {}
  for w, (coef, c) in joined_scores.items():
    if w in model.vocab:
      anchors.append( (coef, model[w] / np.sqrt(np.square(model[w]).sum())))
    anchor_words[w] = coef
    
      
  target_vocab = {}
  reader = codecs.getreader('utf-8')
  with reader(gzip.open(ops.corpus)) as file:
    for line in file:
      info = line.split('\t',2)
      for w in re.findall('\w(?:\w|[-\'])*\w', info[2].lower()):
        target_vocab[w] = 1+target_vocab.get(w,0)
        
  common_words = [w for w,c in target_vocab.items() if c > 100]
  all_words = [(w,coef) for w,coef in anchor_words.items()]
  for w in common_words:
    if w in anchor_words or w not in model.vocab:
      continue
    vec = model[w] / np.sqrt(np.square(model[w]).sum())
    
    neighbs = [(coef, np.exp(np.dot(v, vec))) for coef,v in anchors]
    neighbs = sorted(neighbs, key=lambda x: x[1], reverse=True)[:5]
    coef = sum( y*d for y,d in neighbs) / sum(d for _,d in neighbs)
    all_words.append( (w, coef))

  
  coef_ = np.zeros( (1,len(all_words)))

  y = label_lookup['1']
  n = label_lookup['0']
  all_words = sorted(all_words, key=lambda x: x[0])

  newvocabulary = {}
  newnames = []
  newmodel = classifier
  newdv = dv
  for i,(w,coef) in enumerate(all_words):
    coef_[0,i] = coef
    newnames.append(w)
    newvocabulary[w] = i
    
  newdv.feature_names_ = newnames
  newdv.vocabulary_ = newvocabulary
  newmodel.coef_ = coef_
  

  logging.info('Saving new model in %s.model.gz' % ops.output)
  with gzip.open('%s.model.gz' % ops.output,'wb') as ofd:
    pickle.dump((newmodel, newdv, label_lookup, tokenizer),ofd)
    
methods_nb = {'extract_words': extract_words_nb,
            'make': make_nb}

methods_lrsvm = {'extract_words': extract_words_lrsvm,
            'make': make_lrsvm}

            
if __name__ == '__main__':
  ops = options()
  logging.basicConfig(level=logging.INFO)
  if ops.type == 'naive_bayes':
    methods_nb[ops.method](ops)
  elif ops.type in {'logistic_regression', 'svm'}:
    methods_lrsvm[ops.method](ops)