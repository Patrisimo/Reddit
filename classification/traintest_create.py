# Create the train and test splits

import json
import codecs
import random
import re

def bin(doc, field):
	val = doc[field]
	if field == 'score':
		if val < 0:
			return '-1'
		elif val < 5:
			return '1'
		elif val < 11:
			return '5'
		elif val < 50:
			return '20'
		else:
			return '50'
	elif field == 'gilded':
		if val > 0:
			return 1
		else:
			return 0
	else:	
		return val
with open('reddit_all.json') as file:
	all = json.load(file)

fnames = {'es': 'spanish', 'es_sub': 'spanish-sub',
					'fr': 'french', 'fr_sub': 'french-sub',
					'it': 'italian', 'it_sub': 'italian-sub'}
	
for key, docs in all.items():
	test = set(random.sample(range(len(docs)), len(docs) // 5))
	for id, name in zip(['score', 'gilded', 'controversiality'], ['score','gilded','contro']):
		with codecs.open('%s_%s_train.tsv' % (fnames[key], name), 'w', 'utf-8') as trainfile:
			with codecs.open('%s_%s_test.tsv' % (fnames[key], name), 'w', 'utf-8') as testfile:
				for i, doc in enumerate(docs):
					line = '%s\t%s\t%s\n' % (doc['id'], bin(doc, id), re.sub('\s+', ' ', doc['body']))
					if i in test:
						testfile.write(line)
					else:
						trainfile.write(line)
		print('%s_%s complete' % (fnames[key], name))
	
