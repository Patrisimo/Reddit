# This is naive bayes, but the data is projected onto the top singular values first
# TThis hopefully increases independence of the features, an assumption of Naive Bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import numpy as np
from itertools import chain
from collections import Counter


class SVDBayes(MultinomialNB):
    
    def __init__(self):
        MultinomialNB.__init__(self)
        self.restr_matrix = None
        self.feature_vecs = None
        self.mean = None
        self.shifts = None

    #@override
    def fit(self, X, y, sample_weight=None, k=200):
        # Need to find the singular values and save the singular vectors
        # First, make sure we aren't taking too many singular vectors
        k = min(k, min(X.shape))
        # The data needs to be centered in order to do SVD, but there are too many features to do SVD
        # Kill some of the features, maybe leave 10*k many
        print "Finding top features"
        weights = sorted( enumerate([i for i in (np.ones( (1,X.shape[0]) ) * X).flat]), key=lambda w: w[1], reverse=True)
        # We're transforming X from an M x N to M x 10k matrix, so we need a N x 10k 'identity' matrix
        # This technically would be faster with just column extraction, but 1. we're in a csr and 2. this isn't that slow
        restr = min(10*k, X.shape[1])
        good_features = [ weights[i][0] for i in xrange(restr)]
        restr_matrix = csr_matrix( ([1 for _ in xrange(restr)], (good_features, xrange(restr))), shape=(X.shape[1], restr))
        X_restr = X * restr_matrix
        print "Centering data"
        X_cen, mean = SVDBayes.center(X_restr) # the mean is required to center new data
        proj = X_cen.transpose() * X_cen
        proj = np.array(proj.todense()) # todense gives an np.matrix, but I don't like using those 
        # Now find the eigenvalues of this
        print "Calculating singular values"
        eigvs, evecs = np.linalg.eig(proj)
        eigv_order = sorted(xrange(eigvs.shape[0]), key=lambda n: eigvs[n], reverse=True)
        # evecs is a square matrix, where each column is an eigenvector. Need to extract k many of them
        evec_mat = evecs[:,eigv_order[:k]].real
        # project onto singular vectors
        print "Projecting onto singular vectors"
        new_data = X_cen * evec_mat
        # Now we need to make sure that everything is positive, which I think is just increase each column's values appropriately
        shifts = np.array( [[ new_data[:,i].min() -1 for i in xrange(new_data.shape[1])]]) # -1 just to be safe
        pos_data = new_data - np.dot(np.ones((new_data.shape[0],1)), shifts) # This might not be the most efficient way to do this
        
        print "Preparing to fit Naive Bayes"
        # Now what do I need in order to transform new observations?
        # What features/columns get used
        self.restr_matrix = restr_matrix
        # Where the center of the restricted data is
        self.mean = mean
        # How to transform the centered restricted data onto the singular values
        self.feature_vecs = evec_mat
        # How to adjust everything to be positive
        self.shifts = shifts
        
        print "Running Naive Bayes"
        super(SVDBayes, self).fit(pos_data, y, sample_weight=sample_weight)
        return self

    #@override
    def predict(self, X):
        X = self.prepare(X)
        return super(SVDBayes, self).predict(X)

    #@override
    def predict_proba(self, X):
        X = self.prepare(X)
        return super(SVDBayes, self).predict_proba(X)

    #@override
    def predict_log_proba(self, X):
        X = self.prepare(X)
        return super(SVDBayes, self).predict_log_proba(X)

    #@override
    def score(self, X, y, sample_weight=None):
        X = self.prepare(X)
        return super(SVDBayes, self).score(X,y,sample_weight=sample_weight)

    def prepare(self, X):
        # Transform the new observation
        # Remove bad columns
        X = np.array((X * self.restr_matrix).todense())
        # Center the data
        X = X - np.dot( np.ones((X.shape[0],1)), self.mean)
        # Tranform
        X = X * self.feature_vecs # project onto singular vectors
        # Shift
        X = X - np.dot(np.ones((X.shape[0],1)), self.shifts)
        # Zero out any negatives that are left
        for i in xrange(X.shape[0]):
            for j in xrange(X.shape[1]):
                if X[i,j] < 0:
                    X[i,j] = 0
        
        return X

    @staticmethod
    def center(A):
        length = A.shape[0]
        mat_mean = (csr_matrix(np.ones((length,length)))/length) * A
        return A - mat_mean, mat_mean[0,:].todense()

class TrunSVDBayes(MultinomialNB):
    
    def __init__(self, debug_level):
        MultinomialNB.__init__(self)
        self.tsvd = TruncatedSVD(n_components=debug_level)
        self.shift = [0 for _ in xrange(debug_level)]

    #@override
    def fit(self, X, y, sample_weight=None, k=200):
        print "Running TruncSVD"
        X = self.tsvd.fit_transform(X)
        print "Shifting"
        self.shift = [ min(X[:,i]) for i in xrange(X.shape[1]) ]
        for i in xrange(X.shape[0]):
            for j in xrange(X.shape[1]):
                X[i,j] -= self.shift[j]
        print "Running Naive Bayes"
        super(TrunSVDBayes, self).fit(X, y, sample_weight=sample_weight)
        return self

    #@override
    def predict(self, X):
        X = self.prepare(X)
        return super(TrunSVDBayes, self).predict(X)

    #@override
    def predict_proba(self, X):
        X = self.prepare(X)
        return super(TrunSVDBayes, self).predict_proba(X)

    #@override
    def predict_log_proba(self, X):
        X = self.prepare(X)
        return super(TrunSVDBayes, self).predict_log_proba(X)

    #@override
    def score(self, X, y, sample_weight=None):
        X = self.prepare(X)
        return super(TrunSVDBayes, self).score(X,y,sample_weight=sample_weight)

    def prepare(self, X):
        # Transform the new observation
        X = self.tsvd.transform(X)
        # Zero out any negatives that are left
        for i in xrange(X.shape[0]):
            for j in xrange(X.shape[1]):
                if X[i,j] < self.shift[j]:
                    X[i,j] = 0
                else:
                    X[i,j] -= self.shift[j]
        
        return X


class TrunSVDLR(LogisticRegression):
    
    def __init__(self, debug_level):
        LogisticRegression.__init__(self)
        self.tsvd = TruncatedSVD(n_components=debug_level)
#        self.shift = [0 for _ in xrange(debug_level)]

    #@override
    def fit(self, X, y, sample_weight=None, k=200):
        print "Running TruncSVD"
        X = self.tsvd.fit_transform(X)
#        print "Shifting"
#        self.shift = [ min(X[:,i]) for i in xrange(X.shape[1]) ]
#        for i in xrange(X.shape[0]):
#            for j in xrange(X.shape[1]):
#                X[i,j] -= self.shift[j]
        print "Running Naive Bayes"
        super(TrunSVDLR, self).fit(X, y, sample_weight=sample_weight)
        return self

    #@override
    def predict(self, X):
        X = self.prepare(X)
        return super(TrunSVDLR, self).predict(X)

    #@override
    def predict_proba(self, X):
        X = self.prepare(X)
        return super(TrunSVDLR, self).predict_proba(X)

    #@override
    def predict_log_proba(self, X):
        X = self.prepare(X)
        return super(TrunSVDLR, self).predict_log_proba(X)

    #@override
    def score(self, X, y, sample_weight=None):
        X = self.prepare(X)
        return super(TrunSVDLR, self).score(X,y,sample_weight=sample_weight)

    def prepare(self, X):
        # Transform the new observation
        if X.shape[1] == self.tsvd.components_.shape[1]:
            X = self.tsvd.transform(X)
        # Zero out any negatives that are left
#        for i in xrange(X.shape[0]):
#            for j in xrange(X.shape[1]):
#                if X[i,j] < self.shift[j]:
#                    X[i,j] = 0
#                else:
#                    X[i,j] -= self.shift[j]
        return X

class BayesTest(MultinomialNB):
    
    def __init__(self, debug_level):
        MultinomialNB.__init__(self)
        self.restr_matrix = None
        self.feature_vecs = None
        self.mean = None
        self.shifts = None
        self.debug = debug_level

    #@override
    def fit(self, X, y, sample_weight=None, k=200):
        # Need to find the singular values and save the singular vectors
        # First, make sure we aren't taking too many singular vectors
        k = min(k, min(X.shape))
        # The data needs to be centered in order to do SVD, but there are too many features to do SVD
        # Kill some of the features, maybe leave 10*k many
        # Since this is the debug version, initialize all the stored things to avoid errors
        restr_matrix = None
        feature_vecs = None
        mean = None
        shifts = None

        if self.debug % 2 == 0:
            print "Finding top features"
            weights = sorted( enumerate([i for i in (np.ones( (1,X.shape[0]) ) * X).flat]), key=lambda w: w[1], reverse=True)
            # We're transforming X from an M x N to M x 10k matrix, so we need a N x 10k 'identity' matrix
            # This technically would be faster with just column extraction, but 1. we're in a csr and 2. this isn't that slow
            restr = min(10*k, X.shape[1])
            good_features = [ weights[i][0] for i in xrange(restr)]
            restr_matrix = csr_matrix( ([1 for _ in xrange(restr)], (good_features, xrange(restr))), shape=(X.shape[1], restr))
            X = X * restr_matrix
        
        if self.debug % 3 == 0:
            print "Centering data"
            X, mean = SVDBayes.center(X) # the mean is required to center new data

        if self.debug % 5 == 0:
            proj = X.transpose() * X
            proj = np.array(proj.todense()) # todense gives an np.matrix, but I don't like using those 
            # Now find the eigenvalues of this
            print "Calculating singular values"
            eigvs, evecs = np.linalg.eig(proj)
            eigv_order = sorted(xrange(eigvs.shape[0]), key=lambda n: eigvs[n], reverse=True)
            # evecs is a square matrix, where each column is an eigenvector. Need to extract k many of them
            evec_mat = evecs[:,eigv_order[:k]].real
            # project onto singular vectors
            print "Projecting onto singular vectors"
            X = X * evec_mat

        if self.debug % 11 == 0:
            evec_mat = TruncatedSVD(n_components=k)
            X = evec_mat.fit_transform(X)

        if self.debug % 7 == 0:
            # Now we need to make sure that everything is positive, which I think is just increase each column's values appropriately
            shifts = np.array( [[ X[:,i].min() -1 for i in xrange(X.shape[1])]]) # -1 just to be safe
            X = X - np.dot(np.ones((X.shape[0],1)), shifts) # This might not be the most efficient way to do this
        
            

        print "Preparing to fit Naive Bayes"
        # Now what do I need in order to transform new observations?
        # What features/columns get used
        self.restr_matrix = restr_matrix
        # Where the center of the restricted data is
        self.mean = mean
        # How to transform the centered restricted data onto the singular values
        self.feature_vecs = evec_mat
        # How to adjust everything to be positive
        self.shifts = shifts
        print "Running Naive Bayes"
        super(BayesTest, self).fit(X, y, sample_weight=sample_weight)
        return self

    #@override
    def predict(self, X):
        X = self.prepare(X)
        return super(BayesTest, self).predict(X)

    #@override
    def predict_proba(self, X):
        X = self.prepare(X)
        return super(BayesTest, self).predict_proba(X)

    #@override
    def predict_log_proba(self, X):
        X = self.prepare(X)
        return super(BayesTest, self).predict_log_proba(X)

    #@override
    def score(self, X, y, sample_weight=None):
        X = self.prepare(X)
        return super(BayesTest, self).score(X,y,sample_weight=sample_weight)

    def prepare(self, X):
        # Transform the new observation
        # Remove bad columns
        if self.debug % 2 == 0:
            X = np.array((X * self.restr_matrix).todense())
        # Center the data
        if self.debug % 3 == 0:
            X = X - np.dot( np.ones((X.shape[0],1)), self.mean)
        # Tranform
        if self.debug % 5 == 0:
            X = np.dot(X, self.feature_vecs) # project onto singular vectors
        # Shift
        if self.debug % 11 == 0:
            X = self.feature_vecs.transform(X)
        if self.debug % 7 == 0:
            X = X - np.dot(np.ones((X.shape[0],1)), self.shifts)
            # Zero out any negatives that are left
            for i in xrange(X.shape[0]):
                for j in xrange(X.shape[1]):
                    if X[i,j] < 0:
                        X[i,j] = 0

        return X

    @staticmethod
    def center(A):
        length = A.shape[0]
        mat_mean = (csr_matrix(np.ones((length,length)))/length) * A
        return A - mat_mean, mat_mean[0,:].todense()

    
