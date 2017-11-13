I want s

def process(fname, mused, output, topics=100):
	with codecs.open(mused,'r','utf-8') as file:
		most_used = { w: i for i,w in enumerate(json.load(file))}
	
	wtcf = sparse.lil_matrix( (len(most_used), topics))
  
	with codecs.open(fname,'r','utf-8') as file:
		for i,line in enumerate(file):
			info = line.strip().split()
			word = info[1]
			if word not in most_used:
				continue
      
			for j,v in map(lambda x: x.split(':'), info[2:]):
				wtcf[most_used[word],int(j)] = v
  
	with open('%s' % output, 'wb') as file:
		pickle.dump( wtcf.tocsr(), file)
		
		
from scipy import sparse
import codecs, pickle
def process_full(fname, output, topics=100):
	wtcf = sparse.lil_matrix( (400000, topics))
	vocab = []
	
	with codecs.open(fname,'r','utf-8') as file:
		for i,line in enumerate(file):
			info = line.strip().split()
			word = info[1]
			vocab.append(word)
			
      
			for j,v in map(lambda x: x.split(':'), info[2:]):
				wtcf[i,int(j)] = v
  
	cut = wtcf.tocsr()[:len(vocab)]
	
	with open('%s' % output, 'wb') as file:
		pickle.dump( [vocab, cut], file)				
		
def normalize(mat):
	return sparse.diags( 
		[ 1./ np.sqrt(np.square(mat.getrow(i).data).sum()) 
		if len(mat.getrow(i).data) > 0 else 0 for i in range(mat.shape[0])],
		0) * mat

def slope(fname, lstsq=True, normalize=True, min_ct = 1):
	with open(fname, 'rb') as file:
		vocab, wtcf = pickle.load(file)
	groupings = {i: [] for i in range(100)}
	for i in range(wtcf.shape[0]):
		row = wtcf.getrow(i)
		topics = sum(k >= min_ct for k in row.data)
		ct = sum( k for k in row.data if k >= min_ct)
		if topics > 0:
			groupings[topics].append(ct)
	if normalize:
		total = max(max(l, default=1) for l in groupings.values())
	else:
		total = 1
	avgs = {k: np.mean(v) / total for k,v in groupings.items() if len(v) > 0}
	a_xs = np.log(np.array(list(avgs.values())))
	a_ys = np.log(np.array(list(avgs.keys())))
	A = np.vstack([a_xs, np.ones(len(a_xs))]).T
	if lstsq:
		return np.linalg.lstsq(A, a_ys)
	else:
		return stats.linregress(a_xs, a_ys)
	
		
def zscores(l):
	mu = np.mean(l)
	std = np.std(l)
	print("Mean: %.4f\t Sig: %.4f" % (mu, std))
	print("Within 1: %f" % ( sum( abs( (i-mu) / std) < 1 for i in l) * 1. / len(l)))
	print("Within 2: %f" % ( sum( abs( (i-mu) / std) < 2 for i in l) * 1. / len(l)))
	print("Within 3: %f" % ( sum( abs( (i-mu) / std) < 3 for i in l) * 1. / len(l)))
	
def topic_compare():
	topics = {}
	for i in range(2):
		with open("branch%d_100_full.pkl" % i, 'rb') as file:
			vocab, wtcf = pickle.load(file)
		for i,w in enumerate(vocab):
			if w not in topics:
				topics[w] = []
			topics[w].append(wtcf.getrow(i))
	
	return topics
	
	

def slope(fname, lstsq=True, normalize=True, topics=100):
	groupings = {i: [] for i in range(topics)}

	flag = True
	while flag:
			try:
					with codecs.open(fname, 'r', 'utf-8') as file:
							flag = False
							for line in file:
									info = line.strip().split()
									k = len(info) -2
									n = 0
									for _,v in map(lambda x: x.split(':'), info[2:]):
											n += int(v)
									if n > 0:
											groupings[k].append(n)
			except FileNotFoundError:
					sleep(60)

	if normalize:
			total = max(max(l) if len(l) > 0 else 1 for l in groupings.values())
	else:
			total = 1
	avgs = {k: np.mean(v) / total for k,v in groupings.items() if len(v) > 0}
	a_ns = np.log(np.array(list(avgs.values())))
	a_logks = np.log(np.array(list(avgs.keys())))
	a_ks = np.array(list(avgs.keys()))
	A = np.vstack([a_ks, a_logks]).T
	B = np.vstack([a_ks, a_logks, np.ones( (len(a_ks),))]).T
	
	output = "Data for %s" % (fname)
	
	# linear regression
	L1lstsq.__defaults__ = (B, a_ns)
	linear_info = np.vstack(
		[ optimize.leastsq(L1lstsq, np.array([random.random()*20-10 
		for _ in range(3)]))[0] for _ in range(100)]).T
	diffs = np.square(np.dot(B, linear_info) - np.vstack([a_ns for _ in range(linear_info.shape[1])]).T).sum(0)
	power_inds = [i for i in range(len(diffs)) 
		if diffs[i] < 10 * diffs.min() and 
		linear_info[i,0] < linear_info[i,1]]
	logar_inds = [i for i in range(len(diffs)) 
		if diffs[i] < 10 * diffs.min() and 
		linear_info[i,0] > linear_info[i,1]]
	if len(power_inds) > 0:
		power_slope = np.mean(linear_info[power_inds,1])
		power_int = np.mean(linear_info[power_inds,2])
		power_var = np.var(linear_info[power_inds,1])
	else:
		power_slope = 0
		power_int = 0
		power_var = 0
	if len(logar_inds) > 0:
		logar_slope = np.mean(linear_info[logar_inds,0])
		logar_int = np.mean(linear_info[logar_inds,2])
		logar_var = np.var(linear_info[logar_ids,0])
	else:
		logar_slope = 0
		logar_int = 0
		logar_var = 0
	
	output += "\nLinear regression: logn = %+.4f logk %+.4f k %+.5f" % (linr[0], linr[1], linr[2])
	
	# linear
	L1lstsq.__defaults__ = (A, a_ns)
	linear_info = np.vstack(
		[ optimize.leastsq(L1lstsq, np.array([random.random()*20-10 
		for _ in range(2)]))[0] for _ in range(100)]).T
	diffs = np.square(np.dot(A, linear_info) - np.vstack([a_ns for _ in range(linear_info.shape[1])]).T).sum(0)
	power_inds = [i for i in range(len(diffs)) 
		if diffs[i] < 10 * diffs.min() and 
		linear_info[i,0] < linear_info[i,1]]
	logar_inds = [i for i in range(len(diffs)) 
		if diffs[i] < 10 * diffs.min() and 
		linear_info[i,0] > linear_info[i,1]]
	if len(power_inds) > 0:
		power_slope = np.mean(linear_info[power_inds,1])
		power_var = np.var(linear_info[power_inds,1])
	else:
		power_slope = 0
		power_var = 0
	if len(logar_inds) > 0:
		logar_slope = np.mean(linear_info[logar_inds,0])
		logar_var = np.var(linear_info[logar_ids,0])
	else:
		logar_slope = 0
		logar_var = 0
		
	
	
	
	output += "\nMatrix:            logn = %+.4f logk %+.4f k" % (lst[0][0], lst[0][1])
	print(output)
	
def L1lstsq(x, A=None, y=None):
	diff = np.dot(A,x) - y
	norm = np.sqrt(np.abs(x)).sum()
	return diff * max(norm, 0.1)