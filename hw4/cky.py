from PCFG import PCFG
import math


def load_sents_to_parse(filename):
	sents = []
	with open(filename) as f:
		for line in f:
			line = line.strip()
			if line:
				sents.append(line)
	return sents


def cnf_cky(pcfg, sent):
	### YOUR CODE HERE
	import numpy as np
	def init(n, N, Q):  # need to implement
		pi = {}
		for X in N:
			for i in xrange(n):
				pi[i, i, X] = Q.get((X, (sent[i],)), 0)
		return pi
	
	def get_N(i, j):
		return {}
	
	def get_q(grammar):
		q_dict = {}
		for left, right in grammar._rules.items():
			denom = sum([d[1] for d in right])
			for der in right:
				q_dict[(left, tuple(der[0]))] = der[1] / float(denom)
		return q_dict
	
	def updatePi(pi, i, j, Q, X):
		best_pi = float('-inf')
		best_rule = ()
		for rule in Q.keys():
			if rule[0] != X: continue
			for s in xrange(i,j):
				if len(rule[1]) > 1:
					# curr_pi = Q.get(rule,0.0) * pi.get((i, s, rule[1][0]),0) * pi.get((s+1, j, rule[1][1]),0)
					curr_pi = Q.get(rule, 0.0) * pi[(i, s, rule[1][0])] * pi[(s + 1, j, rule[1][1])]
					if curr_pi > best_pi:
						best_pi = curr_pi
						best_rule = rule
		return best_pi, best_rule
	
	# from Q1:
	# def gentree(rule):
	#     """
	#         Generates a derivation tree from a given symbol
	#     """
	#     ### YOUR CODE HERE
	#     if self.is_terminal(symbol):
	#         return symbol
	#     else:
	#         expansion = self.random_expansion(symbol)
	#         expansion_filled = " ".join(self.gentree(s) for s in expansion)
	#         return "("+symbol+" "+expansion_filled+")"
	
	def get_derivation(bp):
		return "tree"
	
	print(sent)
	# print(pcfg._rules)
	sent = sent.split(' ')
	n = len(sent)
	# N = get_N(0, n)  # get all tags
	Q = get_q(pcfg)
	# print('Q:', Q)
	N = pcfg._rules.keys()
	pi = init(n, N, Q)
	bp = {}
	# print('sums: ', pcfg._sums)
	indices = [ind for gap in xrange(n) for ind in np.ndindex((n,n)) if ind[0]<ind[1] and ind[1]-ind[0] == gap]
	for (i,j) in indices:
			# for X in get_N(i, j):
			for X in N:
				pi_max, pi_arg_max = updatePi(pi, i, j, Q, X)
				pi[i, j, X] = pi_max
				bp[i, j, X] = pi_arg_max
	print 'Pi after: %s' % pi
	print 'BP: %s' % bp
	return get_derivation(bp)
	### END YOUR CODE
	return "FAILED TO PARSE!"


def non_cnf_cky(pcfg, sent):
	### YOUR CODE HERE
	# raise NotImplementedError
	### END YOUR CODE
	return "FAILED TO PARSE!"


if __name__ == '__main__':
	import sys
	
	cnf_pcfg = PCFG.from_file_assert(sys.argv[1], assert_cnf=True)
	non_cnf_pcfg = PCFG.from_file_assert(sys.argv[2])
	sents_to_parse = load_sents_to_parse(sys.argv[3])
	for sent in sents_to_parse:
		print cnf_cky(cnf_pcfg, sent)
		print non_cnf_cky(non_cnf_pcfg, sent)
		break
