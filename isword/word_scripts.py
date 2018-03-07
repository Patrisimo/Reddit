# See if it can be determined whether a given n-gram is a word or not

# What I want is for each word and 2gram the distribution of sentence lengths

# We can start with just average though
import re
import codecs

comment_lengths = {}
sentence_lengths = {}
vocab = {}

with codecs.open('reddit_20news_cleaned.txt','r','utf-8') as file:
	for i,line in enumerate(file):
		print('%d\r' % i, end='')
		info = line.split('\t',1)
		clean = re.sub('[^a-zA-Z_0-9.?! ]','', info[1]).lower()
		sentences = re.findall('[^.?!]+',clean)
		dict_add(comment_lengths, len(re.findall('[a-zA-Z_0-9]+',clean)))
		for sent in sentences:
			words = re.findall('[a-zA-Z_0-9]+',sent)
			dict_add(sentence_lengths, len(words))
			for w in words:
				dict_add(vocab, w)

common_words = [ w for w,c in vocab.items() if c > 39 ]
word_sent_lengths = {w: [] for w in common_words}
				
with codecs.open('reddit_20news_cleaned.txt','r','utf-8') as file:
	for i,line in enumerate(file):
		print('%d\r' % i, end='')
		info = line.split('\t',1)
		clean = re.sub('[^a-zA-Z_0-9.?! ]','', info[1]).lower()
		sentences = re.findall('[^.?!]+',clean)
		for sent in sentences:
			words = re.findall('[a-zA-Z_0-9]+',sent)
			ct = len(words)
			for w in words:
				if w in word_sent_lengths:
					word_sent_lengths[w].append(ct)
				
# and let's do it for some twograms
very_common = { j: i for i,j in enumerate([w for w,c in vocab.items() if c > 300 and c < 300000])}
twograms = {w: {} for w in very_common.values()}
with codecs.open('reddit_20news_cleaned.txt','r','utf-8') as file:
	for i,line in enumerate(file):
		print('%d\r' % i, end='')
		info = line.split('\t',1)
		clean = re.sub('[^a-zA-Z_0-9.?! ]','', info[1]).lower()
		sentences = re.findall('[^.?!]+',clean)
		for sent in sentences:
			words = re.findall('[a-zA-Z_0-9]+',sent)
			ct = len(words)
			word_ids = [ very_common[w] for w in words if w in very_common]
			for i in range(1,len(word_ids)):
				first = word_ids[i-1]
				second = word_ids[i]
				if second in twograms[first]:
					twograms[first][second].append(ct)
				else:
					twograms[first][second] = [ct]

for i in twograms.keys():
	ks = list(twograms[i].keys())
	for j in ks:
		if len(twograms[i][j]) < 5:
			twograms[i].pop(j)
		
# let's clean the data real quick

def plot_word(word):
	


def dict_add(d, val):
	if val in d:
		d[val] += 1
	else:
		d[val] = 1