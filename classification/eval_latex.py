import gzip
from argparse import ArgumentParser
import numpy as np

score = 0
outOf = 0

parser = ArgumentParser()
langs = ['spanish', 'french', 'italian']
variants = ['-sub', '']
tests = ['gilded','contro','score']
algs = ['naive_bayes','logistic_regression']



for alg in algs:
	print(' \\multicolumn{4}{|c|}{%s} \\\\' % (alg))
	print('& %s \\\\' % ('&'.join(s.capitalize() for s in tests)))
	for lang in langs:
		for var in variants:
			print('%s%s ' % (lang.capitalize(), var), end='')
			for test in tests:
				fname = '%s%s_%s_%s.prob.gz' % (lang, var, test, alg)
				try:
					with gzip.open(fname) as file:
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
							
							numIdentified[scoresSorted[0][0]] += 1

								
						#print('Evaluation:')
						#for label, position in sorted(labels.items(), key=lambda x: x[0]):
							#print('%s: %d/%d' % (label, numCorrect[position], numLabels[position]))
						#print('\nOverall prob: %f' % (score / sum(numLabels)))
						if min(numLabels) == 0:
							print('& err', end = '')
						else:
							fscores = [0]*len(numLabels)
							for i in range(len(numLabels)):
								fscores[i] = 2. * max(numCorrect[i], 0.01) / (numLabels[i] + numIdentified[i])
							#print('& %.4f' % ( len(fscores) / sum( 1./f for f in fscores)), end='')
							print('& %d' % min(numLabels), end='')
							
				except IOError as e:
					print('& err', end='')
			print('\\\\')