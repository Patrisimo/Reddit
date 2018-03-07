import gzip
from argparse import ArgumentParser
import numpy as np
from matplotlib import pyplot as plt


def evaluate_wrapper(ops):
  return evaluate(ops.probs)

def roc_wrapper(ops):
  return roc(ops.probs, output=ops.output, plot=ops.plot)

def evaluate(probs):
  score = 0
  with gzip.open(probs) as file:
    header = file.readline().decode().strip().split('\t')
    labels = {j:i for i,j in enumerate(header[2:])}
    numLabels = [0 for _ in labels]
    numCorrect = [0 for _ in labels]
    numIdentified = [0 for _ in labels]
    for line in file:
      data = line.decode().strip().split('\t')
      scores = list(map(float, data[2:]))
      if data[1] not in labels:
        labels[data[1]] = len(labels)
        numLabels.append(0)
        numCorrect.append(0)
        numIdentified.append(0)
      correctIndex = labels[data[1]]
      numLabels[correctIndex] += 1
      #score += np.exp(scores[correctIndex])
      
      scoresSorted = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
      if scoresSorted[0][0] == correctIndex:
        numCorrect[correctIndex] += 1
      else:
        score += (np.exp(scoresSorted[0][1]) - np.exp(scores[correctIndex])) if correctIndex < len(scores) else np.exp(scoresSorted[0][1])
      numIdentified[scoresSorted[0][0]] += 1

        
  print('Evaluation:')
  for label, position in sorted(labels.items(), key=lambda x: x[0]):
    print('%s: %d/%d' % (label, numCorrect[position], numLabels[position]))
  print('\nOverall prob: %f' % (score / sum(numLabels)))
  if len(numLabels) == 2:
    i = 0 if numLabels[0] < numLabels[1] else 1 # Pick the smaller class
    fscore = 2. * numCorrect[i] / (numLabels[i] + numIdentified[i])
    print('F-score: %f' % fscore)
  elif len(numLabels) > 2:
    fscores = [0]*len(numLabels)
    for i in range(len(numLabels)):
      #precision = numCorrect[i] / numIdentified[i]
      #recall = numCorrect[i] / numLabels[i]
      fscores[i] = 2. * max(numCorrect[i], 0.01) / (numLabels[i] + numIdentified[i])
    print('F-score: %f' % ( len(fscores) / sum( 1./f for f in fscores)))

def roc(probs, output=None, plots=False):
  classifications = []
  reader = codecs.getreader('ascii')
  with reader(gzip.open(probs)) as file:
    header = file.readline().strip().split('\t')
    inv_labels = {i:j for i,j in enumerate(header[2:])}
    labels = {j:i for i,j in inv_labels.items()}
    
    for line in file:
      data = line.strip().split('\t')
      scores = list(map(float, data[2:]))
      scoresSorted = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
      classifications.append( (data[1], inv_labels[scoresSorted[0][0]], scores[labels['1']]))
  
  total_yes = 0
  total_no = 0
  true_yes = 0
  false_yes = 0
  rocs = []
  classifications = sorted(classifications, key=lambda x: x[-1], reverse=True)
  for gold, pred, _ in classifications:
    if gold == '1':
      total_yes += 1
      if pred == '1':
        true_yes += 1
    else:
      total_no += 1
      if pred == '1':
        false_yes += 1
    rocs.append( (div(false_yes, total_no), div(true_yes ,total_yes)))
  return rocs

def roc2(probs, output=None, plots=False):
  classifications = []
  reader = codecs.getreader('ascii')
  with reader(gzip.open(probs)) as file:
    header = file.readline().strip().split('\t')
    inv_labels = {i:j for i,j in enumerate(header[2:])}
    labels = {j:i for i,j in inv_labels.items()}
    
    for line in file:
      data = line.strip().split('\t')
      scores = list(map(float, data[2:]))
      scoresSorted = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
      classifications.append( (data[1], inv_labels[scoresSorted[0][0]], scores[labels['1']]))
  
  total_yes = 0
  total_no = 0
  true_yes = 0
  false_yes = 0
  rocs = []
  classifications = sorted(classifications, key=lambda x: x[-1], reverse=True)
  for gold, pred, _ in classifications:
    if gold == '1':
      total_yes += 1
      if pred == '1':
        true_yes += 1
    else:
      total_no += 1
      if pred == '1':
        false_yes += 1
    if total_yes + true_yes + false_yes == 0:
      rocs.append(0)
    else:
      rocs.append( 2. * true_yes / (true_yes + total_yes + false_yes))
  return rocs  
  
def div(a,b):
  if b == 0:
    return 0
  return a / b
  
methods = {'evaluate': evaluate_wrapper,
          'roc': roc_wrapper}
if __name__=='__main__':
  parser = ArgumentParser()
  subparsers = parser.add_subparsers(dest='method', help='evaluation to run')
  parser_roc = subparsers.add_parser('roc')
  parser_roc.add_argument('probs')
  parser_roc.add_argument('-o','--output')
  parser_roc.add_argument('-p','--plot')
  
  parser_ev = subparsers.add_parser('evaluate')
  parser_ev.add_argument('probs')
  ops = parser.parse_args()
  methods[ops.method](ops)