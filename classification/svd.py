from sklearn.decomposition import TruncatedSVD
from scipy import sparse
import numpy as np

class SVDSelect():
  def __init__(self, n_components=100):
    self.n_components=int(n_components)
    self.vocab={}
    self.feature_names=[]
    self.svd=TruncatedSVD(n_components=self.n_components)
    
  def fit_transform(self, data):
    for doc in data:
      for word in doc.keys():
        if word not in self.vocab:
          self.feature_names.append(word)
          self.vocab[word] = len(self.vocab)
    
    mat_data = sparse.lil_matrix( (len(data), len(self.vocab)))
    for i, doc in enumerate(data):
      for word, count in doc.items():
        j = self.vocab[word]
        mat_data[i,j] = count
    reduced = self.svd.fit_transform(mat_data)
    
    if not self.n_components:
      amts = np.cumsum(self.svd.explained_variance_ratio_)
      self.n_components = np.argmax( amts > 0.9) + 1
      reduced = reduced[:,:self.n_components]
    
    return reduced
  
  def transform(self, data):
    mat_data = sparse.lil_matrix( (len(data), len(self.vocab)))
    for i, doc in enumerate(data):
      for word, count in doc.items():
        if word in self.vocab:
          j = self.vocab[word]
          mat_data[i,j] = count
    
    reduced = self.svd.transform(mat_data)[:,:self.n_components]
    return reduced