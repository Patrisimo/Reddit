import gzip
import random
from argparse import ArgumentParser
import codecs
import patrick_algs, evaluate
import json
import logging

writer = codecs.getwriter('utf-8')
reader = codecs.getreader('utf-8')

def options():
  parser = ArgumentParser()
  #parser.add_argument('branch', choices=methods.keys())
  
  subparsers = parser.add_subparsers(help='step to run', dest='branch')
  # pick random
  parser_pr = subparsers.add_parser('pick_random', help='pick an initial random part of the corpus')
  parser_pr.add_argument("input", help='gzipped tsv of entire corpus')
  parser_pr.add_argument('k', type=int, help='Number of documents to select')
  parser_pr.add_argument("output", help='basename for output, will be given appropriate extension')
  
  # train classifier
  parser_tc = subparsers.add_parser('train_classifier', help='train and evaluate a classifier')
  parser_tc.add_argument("input",  help='gzipped tsv of input data')
  parser_tc.add_argument('k', type=int, help='Number of documents to in input')
  parser_tc.add_argument("type",  choices=patrick_algs.models.keys(), help='type of classifier to use')
  parser_tc.add_argument("token",  choices=patrick_algs.tokenizers.keys(), help='tokenizer to use for classifier')
  parser_tc.add_argument("output",  help='basename for output, will be given appropriate extension')
  parser_tc.add_argument("--max_ngram",  type=int, default=4)    
  parser_tc.add_argument("--preproc", dest='preproc', choices=patrick_algs.preprocessors.keys(), default="none", help='preprocess data for classifier')
  parser_tc.add_argument("--preproc-args", dest="preproc_args", nargs='*', default=[], help='arguments for preprocessor')
  
  #pick from classifier
  parser_pc = subparsers.add_parser('pick', help='pick documents based on classifier judgements')
  parser_pc.add_argument("input", help='gzipped tsv')
  parser_pc.add_argument("model", help='gz of previously trained classifier')
  parser_pc.add_argument('k', type=int, help='Number of messages to select')
  parser_pc.add_argument("output", help='basename for output, will be given appropriate extension')
  parser_pc.add_argument("--max_ngram", type=int, default=4)    
  parser_pc.add_argument('--already-seen', help='json of IDs for already-seen messages')
  
  #analyze
  parser_an = subparsers.add_parser('analyze', help='get stats about a dataset')
  parser_an.add_argument("input", help='gzipped tsv')
  
  #prune
  parser_pr = subparsers.add_parser('prune', help='remove negative documents from a dataset')
  parser_pr.add_argument("input", help='gzipped tsv to reduce')
  parser_pr.add_argument("ratio", type=float, help="expected minimum fraction of positive documents")
  parser_pr.add_argument("output", help="basename for output, will be given appropriate extension")
  
  return parser.parse_args()

# 1. Pick a random subset of the messages

def pick_random(ops):
  
  linecount = 0
  chance = ops.k / 200000.
  seen = set()
  with writer(gzip.open('%s.tsv.gz' % ops.output, 'w')) as outfile:
    with reader(gzip.open(ops.input)) as file:
      for line in file:
        linecount += 1
        if random.random() < chance:
          outfile.write(line)
          seen.add(line.split('\t',2)[0])
          ops.k -= 1
        if ops.k == 0:
          break
    if ops.k > 0:
      more = set( random.sample(range(linecount), ops.k))
      with reader(gzip.open(ops.input)) as file:
        for i, line in enumerate(file):
          if i in more:
            outfile.write(line)
            seen.add(line.split('\t',2)[0])
            ops.k -= 1
          if ops.k == 0:
            break
  with open('%s.seen.json' % ops.output, 'w') as file:
    json.dump(list(seen), file)
  print('Random sample saved in %s.tsv.gz, ID set saved in %s.seen.json' % (ops.output, ops.output))
  
def train_classifier(ops):
  # First need to train the classifier
  #def train(output, input, token, modeltype, max_ngram, preproc, preproc_args):
  # Need to split up the data
  test_split = set(random.sample(range(ops.k), ops.k // 5))
  train_split = set()
  for i in range(ops.k):
    if i not in test_split:
      train_split.add(i)
  
  train_args = {'output': ops.output, 
                'input': ops.input, 
                'token': ops.token, 
                'modeltype': ops.type, 
                'max_ngram': ops.max_ngram, 
                'preproc': ops.preproc, 
                'preproc_args': ops.preproc_args,
                'train_split': train_split}
  patrick_algs.train(**train_args)
  
  # Now let's test the algorithm
  #def test(output, input, token, model, max_ngram):
  test_args = {'output': ops.output, 
                'input': ops.input, 
                'model': '%s.model.gz' % ops.output, 
                'max_ngram': ops.max_ngram,
                'test_split': test_split}
  patrick_algs.test(**test_args)
  
  # And evaluate performance
  evaluate.evaluate('%s.prob.gz' % ops.output)
  print('Model saved in %s.model.gz' % ops.output)
            
def pick_from_classifier(ops):
  # I want to pick the k best documents, excluding those already seen
  try:
    with open(ops.already_seen) as file:
      seen = set(json.load(file))
  except (FileNotFoundError, TypeError):
    logging.warning('already seen json file not found')
    seen = set()
  # First, run the classifier over everything
  test_args = {'output': '%s.all' % ops.output,
                'input': ops.input,
                'model': ops.model,
                'max_ngram': ops.max_ngram}
  patrick_algs.test(**test_args)
  comments = {}
  seen_scores = []
  with gzip.open('%s.all.prob.gz' % ops.output) as file:
    header = file.readline().decode().strip().split('\t')
    labels = {j:i for i,j in enumerate(header[2:])}
    for line in file:
      data = line.decode().strip().split('\t')
      if data[0] not in seen:
        score = float(data[2+labels['1']])
        comments[data[0]] = (score, data[1])
      else:
        seen_scores.append((float(data[2+labels['1']]), data[1]))
  
  comment_scores = sorted(comments.items(), key=lambda x: x[1][0], reverse=True)[:ops.k]
  k_best = set( id for id,_ in comment_scores)
  seen_scores = sorted(seen_scores, key=lambda x: x[0], reverse=True)
  lowest_score = comment_scores[-1][1][0]
  repeats = 0
  repeat_pos = 0
  for score, gold in seen_scores:
    if score > lowest_score:
      repeats += 1
      if gold == '1':
        repeat_pos += 1
      if repeats == ops.k:
        break
  
  replaced = 0
  for _, (_, gold) in comment_scores[-repeats:]:
    if gold == '1':
      replaced += 1
  
  print('Avoided %d repeat comments, including %d controversial ones' % (repeats, repeat_pos))
  print('Including those would have supplanted %d controversial comments' % replaced)
  
  written = 0
  with writer(gzip.open('%s.tsv.gz' % ops.output, 'wb')) as outfile:
    with writer(gzip.open('%s.seen.tsv.gz' % ops.output, 'wb')) as seenfile:
      with reader(gzip.open(ops.input)) as file:
        for line in file:
          data = line.split('\t',2)
          if data[0] in k_best:
            outfile.write(line)
            seenfile.write(line)
            seen.add(data[0])
            written += 1
          elif data[0] in seen:
            seenfile.write(line)
  print('%d documents saved' % written)
  with open('%s.seen.json' % ops.output, 'w') as file:
    json.dump(list(seen), file)
  print('New sample saved in %s.tsv.gz, seen ID set updated and saved in %s.seen.json. All seen documents can be found in %s.seen.tsv.gz' % (ops.output, ops.output, ops.output))

def analyze(ops):
  yes = 0
  no = 0
  with reader(gzip.open(ops.input)) as file:
    for line in file:
      data = line.split('\t',2)
      if data[1] == '1':
        yes += 1
      else:
        no += 1
  print('%d total documents: %d yes, %d no' % (yes+no, yes, no))

def prune(ops):
  yes = []
  no = []
  with reader(gzip.open(ops.input)) as file:
    for line in file:
      data = line.split('\t',2)
      if data[1] == '1':
        yes.append(line)
      else:
        no.append(line)
  no_sample = random.sample(no, min(len(no), int(len(yes) * (1. - ops.ratio)/ops.ratio)))
  logging.info('Saving %d yes and %d no (%d total) to %s.tsv.gz' % (len(yes), len(no_sample), len(yes) + len(no_sample), ops.output))
  with writer(gzip.open('%s.tsv.gz' % ops.output, 'w')) as file:
    for line in yes:
      file.write(line)
    for line in no_sample:
      file.write(line)
  
methods = {'pick': pick_from_classifier,
            'pick_random': pick_random,
            'train_classifier': train_classifier,
            'analyze': analyze,
            'prune': prune}
            
if __name__=='__main__':
  ops = options()
  logging.basicConfig(level=logging.INFO)
  methods[ops.branch](ops)