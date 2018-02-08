from sklearn.feature_extraction import DictVectorizer
from collections import deque
import numpy as np
from itertools import chain
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix, hstack
import codecs
import gzip
import re

def write_probabilities(data, output_file):
	# Taken from steamroller.tools.io
	writer = codecs.getwriter("utf-8")
	"""
	data = {"ID" : ("GOLD", {"LAB1" : logprob, ... })
	}
	"""
	codes = set()
	for i, (gold, probs) in data.items():
			for l in probs.keys():
					codes.add(l)
	codes = sorted(codes)
	with writer(gzip.open(output_file, "w")) as ofd:
			ofd.write("\t".join(["DOC", "GOLD"] + codes) + "\n")
			for cid, (label, probs) in data.items():
					ofd.write("\t".join([cid, label] + [str(probs.get(c, float("-inf"))) for c in codes]) + "\n")


def read_data(file, none):
	with codecs.open(file, 'r', 'utf-8') as infile:
		for line in infile:
			yield line.split('\t',2)

def extract_character_ngrams(text, n):
    stack = ["NULL" for _ in xrange(n)]
    ngrams = {}
    for c in text:
        stack = stack[1:]
        stack.append(c)
        ngrams[tuple(stack)] = ngrams.get(tuple(stack), 0) + 1
    return list(ngrams.iteritems())

def extract_bow(text, n=None):
    bow = {}
    words = re.findall("(\w(?:\w|[-_'])*\w)", text.lower())
    for w in words:
        bow[w] = bow.get(w, 0) + 1
    return list(bow.items())


def extract_word_ngrams(text, n):
    stack = ["NULL" for _ in xrange(n)]
    ngrams = {}
    words = filter(lambda c: c.isalnum() or c in ' -', text.lower()).split()
    for w in words:
        stack = stack[1:]
        stack.append(w)
        ngrams[tuple(stack)] = ngrams.get(tuple(stack),0) + 1
    return list(ngrams.iteritems())

def extract_hybrid(text, n):
    stack = ["NULL" for _ in xrange(n)]
    ngrams = {}
    words = filter(lambda c: c.isalnum() or c in ' -', text.lower()).split()
    for w in words:
        if len(w) > n:
            ngrams[tuple(w)] = ngrams.get(tuple(w),0) + 1
    for c in text:
        stack = stack[1:]
        stack.append(c)
        ngrams[tuple(stack)] = ngrams.get(tuple(stack), 0) + 1
    return list(ngrams.iteritems())

def dict_sum(dicts):
  output = {}
  for d in dicts:
    for k,v in d.items():
      output[k] = v + output.get(k,0)
  return output

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

    def fit_transform(self, instances, max_size=1): #returns a grouped version of the data as a sparse matrix
        # Instances is a list of counters, instances[k] tells the count of n-grams from document k
        # Let's make a tree
        # The largest ngrams will generate the smaller ones, so we'll put them at the top
        print("Collecting data")
        total_grams = dict_sum(instances) # This is the total count of the n-gram
        
        queue = deque( NgramGlue.Node(gram, count) for gram,count in total_grams.items() )
        root = NgramGlue.Node('', -1)
        root.left_edges_out = [ n for n in queue ]

        nodes = { n.name: n for n in queue }
        # Build ngram tree
        print("Building ngram tree")
        while len(queue) > 0:
            node = queue.pop()
            if len(node.name) <= 1:
                continue
            children = [ node.name[:-1], node.name[1:]]
            for side,c in enumerate(children):
                if c in nodes: # is an n-gram that exists, and we haven't seen it before
                    cnode = nodes[c]
                else:
                    cnode = NgramGlue.Node(c, 0)
                    nodes[c] = cnode
                    queue.appendleft(cnode)
                if side == 0:
                    node.left_edges_out.add(cnode)
                    cnode.left_edges_in.add(node)
                else:
                    cnode.count += node.count
                    node.right_edges_out.add(cnode)
                    cnode.right_edges_in.add(node)
        # compute scores
        print("Computing scores")
        total_count = sum(total_grams.values())
        for node in nodes.values():
            if node.count == 0:
              nodes.pop(node.name)
              for n in node.left_edges_in:
                n.left_edges_out.remove(node)
              for n in node.right_edges_in:
                n.right_edges_out.remove(node)
              continue
            node.logp = np.log(node.count) - np.log(total_count)

        # Now we've set up our graph, time to compute the glue and select dominant nodes
        print("Computing glue")
        root.glue = -1
        for node in nodes.values():
            cond_prob = []
            for pivot in xrange(1, len(node.name)):
                left = nodes.get( node.name[:pivot], NgramGlue.Node('',1))
                right = nodes.get( node.name[pivot:], NgramGlue.Node('',1))
                cond_prob.append( left.logp + right.logp )
            denom = sum([np.exp(c) for c in cond_prob])
            if denom == 0:
                node.logglue = node.logp
            else:
                denom /= len(cond_prob)
                node.logglue = 2*node.logp - np.log(denom)
            assert np.isfinite(node.logglue)

        # Now to pick the dominant nodes
        print("Selecting dominants")
        queue = deque([root])
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
            node.candidate = False
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

class Truncator():
    def __init__(self, n_components, shift=True):
        n_components = int(n_components)
        self.tsvd = TruncatedSVD(n_components = n_components)
        self.shift = shift
        self.dv = DictVectorizer(sparse=True)

    def fit_transform(self, instances):
        X = self.dv.fit_transform(instances)
        X = self.tsvd.fit_transform(X)
        X = self.sparsify(X)

        return X

    def sparsify(self, X, cutoff=0.5):
        if self.shift:
            neg_axes = csr_matrix( X.shape )
        for i in xrange(X.shape[0]):
            for j in xrange(X.shape[1]):
                if np.abs(X[i,j]) < cutoff:
                    X[i,j] = 0
                if self.shift and X[i,j] < 0:
                    neg_axes[i,j] = -X[i,j]
                    X[i,j] = 0
        X = csr_matrix(X)
        if self.shift:
            X = hstack( (X, neg_axes) )
        return X

    def transform(self, instances):
        X = self.dv.transform(instances)
        X = self.tsvd.transform(X)
        X = self.sparsify(X)
        return X

