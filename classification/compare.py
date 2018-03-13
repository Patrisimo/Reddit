import gzip
import random
from argparse import ArgumentParser
import codecs
import patrick_algs, evaluate
import json
import logging
import numpy as np

writer = codecs.getwriter('utf-8')
reader = codecs.getreader('utf-8')

# I want this to take a json of parameters and train a bunch of algorithms

def options():
  parser = ArgumentParser()
  #parser.add_argument('branch', choices=methods.keys())

  parser.add_argument("input",  help='gzipped tsv of input data')
  parser.add_argument('k', type=int, help='Number of documents to input')
  # parser_tc.add_argument("type",  choices=patrick_algs.models.keys(), help='type of classifier to use')
  # parser_tc.add_argument("token",  choices=patrick_algs.tokenizers.keys(), help='tokenizer to use for classifier')
  parser.add_argument("output",  help='basename for output, will be given appropriate extension')
  # parser_tc.add_argument("--max_ngram",  type=int, default=4)    
  # parser_tc.add_argument("--preproc", dest='preproc', choices=patrick_algs.preprocessors.keys(), default="none", help='preprocess data for classifier')
  # parser_tc.add_argument("--preproc-args", dest="preproc_args", nargs='*', default=[], help='arguments for preprocessor')
  parser.add_argument('--config', help='json of run data', default='config.json')
  
  return parser.parse_args()

def train_classifier(ops):
  # First need to train the classifier
  #def train(output, input, token, modeltype, max_ngram, preproc, preproc_args):
  # Need to split up the data
  # run is {'folds': n, 'name':, 'token':, 'modeltype':, 'max_ngram':, 'preproc':, 'preproc_args':}
  
  test_split = set(random.sample(range(ops.k), ops.k // 5))
  train_split = set()
  for i in range(ops.k):
    if i not in test_split:
      train_split.add(i)
  
  with open(ops.config) as file:
    runs = json.load(file)
    
    
  data = {}  
  for i in range(runs['folds']):    
    test_split = set(random.sample(range(ops.k), ops.k // 5))
    train_split = set()
    for n in range(ops.k):
      if n not in test_split:
        train_split.add(n)
    for run in runs['data']:
      logging.info('Running %s' % run['name'])
      if run['name'] not in data:
        data[run['name']] = []
      run_info={}
      train_args = {'output': '%s_%s_%d' % (ops.output, run['name'], i),
                'input': ops.input, 
                'token': run.get('token', None), 
                'modeltype': run.get('modeltype', None), 
                'max_ngram': run.get('max_ngram', None), 
                'preproc': run.get('preproc', None), 
                'preproc_args': run.get('preproc_args', None),
                'train_split': train_split}
      patrick_algs.train(**train_args)
  
  # Now let's test the algorithm
  #def test(output, input, token, model, max_ngram):
      test_args = {'output': '%s_%s_%d' % (ops.output, run['name'], i),
                'input': ops.input, 
                'token': run.get('token', None), 
                'model': '%s_%s_%d.model.gz' % (ops.output, run['name'], i),
                'max_ngram': run.get('max_ngram', None),
                'test_split': test_split}
      patrick_algs.test(**test_args)
  
  # And evaluate performance
      print('Model saved in %s_%s_%d.model.gz' % (ops.output, run['name'], i))
      eval_info = evaluate.evaluate('%s_%s_%d.prob.gz' % (ops.output, run['name'], i))
      roc_info = evaluate.roc('%s_%s_%d.prob.gz' % (ops.output, run['name'], i))
      run_info['fscore'] = eval_info['fscore']
      run_info['accuracies'] =  [ (nc, nl) for nc, nl in zip(eval_info['correct'], eval_info['labels'])]
      run_info['roc'] = roc_info
      run_info['area'] = area(roc_info)
      data[run['name']].append(run_info)
  
  for i in range(runs['folds']):
    print('Fold %i' % (i+1))
    print('%7s|%s' % (' ', '|'.join( '%7s' % name[:8] for name in data)))
    for name in ['fscore', 'area']:
      print('%7s|' % name, end='')
      for run_data in data.values():
        print('%7f|' % run_data[i][name], end='')
      print('')
      
  print('Average')
  print('%7s|%s' % (' ', '|'.join( '%7s' % name[:8] for name in data)))
  for name in ['fscore', 'area']:
    print('%7s|' % name, end='')
    for run_data in data.values():
      print('%7f|' % np.mean([run_data[i][name] for i in range(runs['folds'])]), end='')
    print('')
    
  
  with open('%s.json' % ops.output, 'w') as file:
    json.dump(data, file)
def area(data):
  data = sorted(data, key=lambda x: x[0])
  area = 0
  for i in range(1,len(data)):
    x0, y0 = data[i-1]
    x1, y1 = data[i]
    area += (y1+y0) * (x1-x0) / 2.
    
  return area

           
if __name__=='__main__':
  ops = options()
  logging.basicConfig(level=logging.INFO)
  train_classifier(ops)