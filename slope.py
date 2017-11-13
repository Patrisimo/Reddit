from scipy import sparse, stats, optimize
from argparse import ArgumentParser
import codecs
import numpy as np
from time import sleep
import random
#from multiprocessing import Pool

def slope(fname, lstsq=True, normalize=True, topics=100, norm=1):
	groupings = {i: [] for i in range(topics)}
	total = 0
	tote = 0
	largest = 0
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
									if n > largest:
											largest = n
									tote += n
			except IOError:
					sleep(60)



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
	output += "\nTotal %d tokens, most common occured %d times" % (tote, largest)
	# linear regression
	L1lstsq.__defaults__ = (B, a_ns, norm)
	linear_info = np.vstack(
		[ optimize.leastsq(L1lstsq, np.array([random.random()*20-10 
		for _ in range(3)]))[0] for _ in range(100)]).T
	diffs = np.square(np.dot(B, linear_info) - np.vstack([a_ns for _ in range(linear_info.shape[1])]).T).sum(0)
	linear_res = np.vstack([linear_info, diffs]).T
	
	# linear
	L1lstsq.__defaults__ = (A, a_ns, norm)
	matrix_info = np.vstack(
		[ optimize.leastsq(L1lstsq, np.array([random.random()*20-10 
		for _ in range(2)]))[0] for _ in range(100)]).T
	diffs = np.square(np.dot(A, matrix_info) - np.vstack([a_ns for _ in range(linear_info.shape[1])]).T).sum(0)
	matrix_res = np.vstack([matrix_info, diffs]).T
	
	return linear_res, matrix_res
	
def L1lstsq(x, A=None, y=None, n=1):
	diff = np.dot(A,x) - y
	norm = np.power(np.power(np.abs(x), n).sum(), 1./n)
	return np.insert(diff,0,norm)
	
def main():
		parser = ArgumentParser()
		parser.add_argument('file', help='Include %d format for where the document count goes')
		parser.add_argument('topics', type=int)
		parser.add_argument('norm', type=float)
		parser.add_argument('output')
		ops = parser.parse_args()
		
		result_lin = np.zeros( (90, 100, 4))
		result_mat = np.zeros( (90, 100, 3))
		for i in xrange(90):
			print '%d\r' % i ,
			n = 5000 + 5000*i
			result_lin[i], result_mat[i] = slope(ops.file % n, topics=ops.topics, norm=ops.norm)
		print ''	
		np.savez(ops.output, linear=result_lin, matrix=result_mat)
		

if __name__=='__main__':
    main()
	