#!/usr/bin/env python

import gzip
from sklearn import naive_bayes, linear_model, svm, metrics, ensemble, dummy
from sklearn.feature_extraction import DictVectorizer
import pickle
import codecs
import logging
from itertools import chain
#from steamroller.tools.io import read_data, write_probabilities, writer, reader, extract_character_ngrams
#from steamroller.tools.io import read_data, write_probabilities, writer, reader

#import svd_bayes
from patrick_tools import read_data, write_probabilities
from patrick_tools import extract_bow, extract_word_ngrams, extract_hybrid, NgramGlue, Truncator

from patrick_tools import extract_character_ngrams

models = {
    "naive_bayes" : (naive_bayes.MultinomialNB, {}, []),
    "svm" : (svm.LinearSVC, {}, ["C"]),
    "logistic_regression" : (linear_model.LogisticRegression, {"class_weight" : "balanced"}, ["C"]),
    "random_forest" : (ensemble.RandomForestClassifier, {"class_weight" : "balanced"}, []),
    "prior" : (dummy.DummyClassifier, {"strategy" : "prior"}, []),
}
tokenizers = {
    "ngram" : (extract_character_ngrams),
    "bow" : (extract_bow),
    "wordgram" : (extract_word_ngrams),
    "hybrid" : (extract_hybrid),
}
preprocessors = {
    "none" : (DictVectorizer),
    "glue" : (NgramGlue),
    "trunc" : (Truncator),
}    


if __name__ == "__main__":

    import argparse

    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input")
    parser.add_argument("--type", dest="type", choices=models.keys())
    parser.add_argument("--train", dest="train")
    parser.add_argument("--test", dest="test")
    parser.add_argument("--model", dest="model")
    parser.add_argument("--output", dest="output")
    parser.add_argument("--tokens", dest="token", choices=tokenizers.keys())
    parser.add_argument("--max_ngram", dest="max_ngram", type=int, default=4)    
    parser.add_argument("--debug-level", dest="debug_level", type=int, default=0)
    parser.add_argument("--preproc", dest="preproc", choices=preprocessors.keys(), default="none")
    parser.add_argument("--preproc-args", dest="preproc_args", nargs='*', default=[])
    options = parser.parse_args()
    if len(options.preproc_args) % 2 != 0:
        raise Exception('Preproc args must be in format "key value"')
        
    
    # training
    if options.train and options.output and options.input:
        instances, labels = [], []
        tokenizer = tokenizers[options.token]
        for cid, label, text in read_data(options.input, options.train):
            instances.append(dict(tokenizer(text, options.max_ngram)))
            labels.append(label)

        dv_class = preprocessors[options.preproc]
        args = { options.preproc_args[2*i]: options.preproc_args[2*i+1] for i in range(len(options.preproc_args)//2)}
        dv = dv_class(**args)

        X = dv.fit_transform(instances)
        label_lookup = {}
        classifier_class, args, hypers = models[options.type]
        classifier = classifier_class(**args)
        for l in labels:
            label_lookup[l] = label_lookup.get(l, len(label_lookup))
        logging.info("Training with %d instances, %d labels", len(instances), len(label_lookup))
        classifier.fit(X, [label_lookup[l] for l in labels])
        with gzip.open(options.output, "w") as ofd:
            pickle.dump((classifier, dv, label_lookup), ofd)            
    # testing
    elif options.test and options.model and options.output and options.input:
        with gzip.open(options.model) as ifd:
            classifier, dv, label_lookup = pickle.load(ifd)
        instances, gold = [], []
        tokenizer = tokenizers[options.token]
        for cid, label, text in read_data(options.input, options.test):
            instances.append(dict(tokenizer(text, options.max_ngram)))
            gold.append((cid, label))
        logging.info("Testing with %d instances, %d labels", len(instances), len(label_lookup))
        X = dv.transform(instances)
        inv_label_lookup = {v : k for k, v in label_lookup.items()}
        data = {}
        order = [inv_label_lookup[i] for i in range(len(inv_label_lookup))]
        if hasattr(classifier, "predict_log_proba"):
            for probs, (cid, g) in zip(classifier.predict_log_proba(X), gold):
                data[cid] = (g, {k : v for k, v in zip(order, probs.flatten())})
        else:
            for pred, (cid, g) in zip(classifier.predict(X), gold):
                probs = [0.0 if i == pred[0] else float("-inf") for i in range(len(order))]
                data[cid] = (g, {k : v for k, v in zip(order, probs)})
        write_probabilities(data, options.output)
    else:
        print("ERROR: you must specify --input and --output, and either --train or --test and --model!")
