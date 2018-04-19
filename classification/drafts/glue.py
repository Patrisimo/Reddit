from sklearn.feature_extraction import DictVectorizer
from collections import deque
import numpy as np
from itertools import chain
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix, hstack
import codecs
import gzip
import re

class NgramGlue():
  class Node():
    def __init__(self, name, count):
      self.name = name
      self.count = count
      self.logp = -np.infty
      self.logglue = np.nan
      self.left_edges_out = set()
      self.right_edges_out = set()
      self.left_edges_in = set()
      self.right_edges_in = set()
      self.dominant = False
      self.candidate = True
      self.components = 1

    def __eq__(self, other):
      return self.name == other.name

    def __hash__(self):
      return self.name.__hash__()

    def parents(self, plist=None):
      if plist is None:
        plist = []
      if len(self.right_edges_in) == 0:
        plist.append( self )
        return plist
      else:
        for p in self.right_edges_in:
          plist = p.parents(plist)
        return plist

  def __init__(self):
    self.ngrams2dom = None
    self.dv = DictVectorizer(sparse='true')

  def __build_tree(total_grams):
    queue = deque( NgramGlue.Node(gram, count) for gram,count in total_grams.items() ) # All the n-grams
    root = NgramGlue.Node('', 0)
    

    nodes = { n.name: n for n in queue }
    # Build ngram tree
    print("Building ngram tree")
    # For each ngram, attach it to its parents (substrings)
    while len(queue) > 0:
      node = queue.pop()
      if len(node.name) == 1:
        root.right_edges_out.add(node)
        node.left_edges_in.add(root)
        root.count += node.count
        continue
      else:
        parents = [ node.name[:-1], node.name[1:]]
        
      
      for side,parent in enumerate(parents):
        if parent in nodes: # is an n-gram that exists, and we haven't seen it before
          parentnode = nodes[parent]
        else:
          parentnode = NgramGlue.Node(parent, 0)
          nodes[parent] = parentnode
          queue.appendleft(parentnode)
        if side == 0:
          node.left_edges_in.add(parentnode)
          parentnode.right_edges_out.add(node)
        else:
          parentnode.count += node.count
          node.right_edges_in.add(parentnode)
          parentnode.left_edges_out.add(node)
    return root, nodes
  def __compute_scores(root, nodes):
    print("Computing scores")
    total_count = root.count
    queue = deque(root.right_edges_out)
    while len(queue) > 0:
      node = queue.pop()
      for n in node.right_edges_out:
        queue.appendleft(n)
      if node.count == 0:
        nodes.pop(node.name)
        for n in node.left_edges_in:
          n.right_edges_out.remove(node)
        for n in node.right_edges_in:
          n.left_edges_out.remove(node)
        continue
      node.logp = np.log(node.count) - np.log(total_count)
  def __compute_glue(root, nodes):
    print("Computing glue")
    root.logglue = -np.inf
    queue = deque(root.right_edges_out)
    while len(queue) > 0:
      node = queue.pop()
      for n in node.right_edges_out:
        queue.appendleft(n)
      if len(node.right_edges_out) == 0:
        continue
      cond_prob = [(-np.inf,0)]
      adj = node.count/root.count
      for pivot in xrange(1, len(node.name)):
        left = nodes.get( node.name[:pivot], NgramGlue.Node('',1))
        if not left.dominant:
          continue
        right = nodes.get( node.name[pivot:], NgramGlue.Node('',1))
        indep_prob = np.log( max(np.exp(left.logglue) - adj, 1/root.count) * max(np.exp(right.logglue) - adj,1/root.count))
        cond_prob.append( ((indep_prob)/(right.components+1), right.components+1) )
      best = sorted(cond_prob, key=lambda x: x[0] + np.log(x[1]), reverse=True)[0]
      if node.logp > best[0] + np.log(best[1]):
        node.dominant = True
        node.logglue = node.logp
      else:
        node.logglue = best[0]
        node.components = best[1]
        
  def __select_dominants(root):
    print("Selecting dominants")
    queue = deque([])
    for n in chain(root.left_edges_out, root.right_edges_out):
      queue.appendleft(n)
      n.candidate = False
    dominants = []
    while len(queue) > 0:
      node = queue.pop()
      for n in chain(node.left_edges_out, node.right_edges_out):
        queue.appendleft(n)
      if not node.candidate:
        continue

      surrounding_glues = [ n.logglue for n in chain(node.left_edges_in, node.left_edges_out, node.right_edges_in, node.right_edges_out) ]
      if len(surrounding_glues) == 0 or node.logglue >= max(surrounding_glues):
        node.dominant = True
        dominants.append(node)
        for n in chain(node.left_edges_in, node.left_edges_out, 
            node.right_edges_in, node.right_edges_out):
          n.candidate = False
      else:
        node.candidate = False
    
  def fit_transform(self, instances, max_size=1): #returns a grouped version of the data as a sparse matrix
    # Instances is a list of counters, instances[k] tells the count of n-grams from document k
    # Let's make a tree
    # The largest ngrams will generate the smaller ones, so we'll put them at the top
    
    print("Collecting data")
    total_grams = dict_sum(instances) # This is the total count of the n-gram
    root, nodes = NgramGlue.__build_tree(total_grams)
    # compute scores
    NgramGlue.__compute_scores(nodes, total_grams)
    # Now we've set up our graph, time to compute the glue and select dominant nodes
    NgramGlue.__compute_glue(root, nodes)
    # Now to pick the dominant nodes
    NgramGlue.__select_dominants(root)
    # Now to only select the dominant features
    print("Reducing data")
    
    # One benefit of this method is that I can deal with new ngrams when
    # it comes time to test. However, those are 1. pretty rare and 2. maybe not as relevant 
    # So we'll just vectorize the entire ngram set and then reduce that
    ngram_count = self.dv.fit_transform(instances)
    # This is a docs x ngram matrix, so I need a ngram x dom matrix to transform it
    # The element (ngram, dom) will indicate how often dom shows up in ngram
    padres = []
    for d in dominants:
      d_pars = d.parents()
      padres.append({})
      for p in d_pars:
        padres[-1][p.name] = 1 + padres[-1].get(p.name, 0)
    ngram2dom = self.dv.transform(padres).transpose()
    
    
    self.ngram2dom = ngram2dom
    
    
    return ngram_count * ngram2dom

  def transform(self, instances):
    X = self.dv.transform(instances)
    return X * self.ngram2dom

  def left_entropy(node1, node2=None, nodes={}, k=1):
    if node2 is None:
      node2 = NgramGlue.Node(('NONE',),0)
    
    count = node1.count - node2.count + k
    entropy = 0
    for neighb in node1.left_edges_out:
      other_name = neighb.name[:1] + node2.name
      if other_name in nodes:
        neighb_count = neighb.count - nodes[other_name].count + k/len(node1.left_edges_out)
      else:
        neighb_count = neighb.count + k/len(node1.left_edges_out)
      neighb_count /= count
      entropy -= neighb_count * np.log(neighb_count)
    
    return entropy
    
  def right_entropy(node1, node2=None, nodes={}, k=1):
    if node2 is None:
      node2 = NgramGlue.Node(('NONE',),0)
    
    count = node1.count - node2.count + k
    entropy = 0
    for neighb in node1.right_edges_out:
      other_name = node2.name + neighb.name[1:]
      if other_name in nodes:
        neighb_count = neighb.count - nodes[other_name].count + k/len(node1.right_edges_out)
      else:
        neighb_count = neighb.count + k/len(node1.right_edges_out)
      neighb_count /= count
      entropy -= neighb_count * np.log(neighb_count)
    
    return entropy
    
  def compare_entropy(tokens, nodes, k=1):
    if type(tokens) == str:
      tokens = tuple(gram.split())
    left = nodes[(tokens[0],)]
    right = nodes[(tokens[-1],)]
    joint = nodes[tuple(tokens)]
    return (left_entropy(left,joint,nodes,k) + right_entropy(right, joint, nodes,k)) / (left_entropy(left,k=k) + right_entropy(right,k=k))
    # print('%.2f  %s  %.2f' % (left_entropy(joint), '-'.join(joint.name), right_entropy(joint)))
    # print('%.2f *%s %s* %.2f' % (left_entropy(left, joint, nodes), ''.join(left.name), ''.join(right.name), right_entropy(right, joint, nodes)))
    # print('%.2f  %s %s  %.2f' % (left_entropy(left), ''.join(left.name), ''.join(right.name), right_entropy(right)))
    