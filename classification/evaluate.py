import gzip
from argparse import ArgumentParser
import numpy as np


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
    fscore = 2. * max(numCorrect[i], 0.01) / (numLabels[i] + numIdentified[i])
    print('F-score: %f' % fscore)
  elif len(numLabels) > 2:
    fscores = [0]*len(numLabels)
    for i in range(len(numLabels)):
      #precision = numCorrect[i] / numIdentified[i]
      #recall = numCorrect[i] / numLabels[i]
      fscores[i] = 2. * max(numCorrect[i], 0.01) / (numLabels[i] + numIdentified[i])
    print('F-score: %f' % ( len(fscores) / sum( 1./f for f in fscores)))

if __name__=='__main__':
  parser = ArgumentParser()
  parser.add_argument('probs')
  ops = parser.parse_args()
  evaluate(ops.probs)