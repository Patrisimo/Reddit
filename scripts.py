# First, create a vocab
# Assume I have it

import json, pickle
import numpy as np


x = np.zeros((20000,20000))


import pickle
from scipy import sparse
import numpy as np
"""Read in a wtcf pickle, and return a matrix of smoothed thingies"""
def make_wtcf(pname, vocab):
  with open(pname, 'rb') as file:
    wtcf = pickle.load(file)
  
  smoothed = sparse.lil_matrix(( len(wtcf['vocab']), wtcf['wtcf'][0].shape[0]))
  for i in range(smoothed.shape[0]):
    row = wtcf['wtcf'][i]
    total = row.sum() + smoothed.shape[1]
    smoothed[i] = row / total
  smoothed = smoothed.tocsc()
  bests = [[vocab[k] for k,_ in sorted(enumerate(smoothed.getcol(i).toarray()), key=lambda x: -x[1])[:20]] for i in range(smoothed.shape[1])]
  return smoothed, bests