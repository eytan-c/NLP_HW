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
            for s in xrange(i, j):
                if len(rule[1]) > 1:
                    # curr_pi = Q.get(rule,0.0) * pi.get((i, s, rule[1][0]),0) * pi.get((s+1, j, rule[1][1]),0)
                    curr_pi = Q.get(rule, 0.0) * pi[(i, s, rule[1][0])] * pi[(s + 1, j, rule[1][1])]
                    if curr_pi > best_pi:
                        best_pi = curr_pi
                        best_rule = (rule, s)
        return best_pi, best_rule

    def get_derivation(bp):
        tree = ''

        def expand(index, symbol):
            if bp.get((index[0], index[1], symbol)) is None:
                print("expend: index, symbol, NO bp ", index, symbol)
                if index[0] == index[1]:
                    return sent[index[0]]
                return -1
            print("expend: index, symbol, bp ", index, symbol, bp.get((index[0], index[1], symbol)))
            rule, s = bp[index[0], index[1], symbol]
            Y, Z = rule[1]
            if pcfg.is_terminal(symbol):
                return symbol
            else:
                Y_expand = expand((index[0], s), Y)
                Z_expand = expand((s + 1, index[1]), Z)
                if Y_expand == -1 or Z_expand == -1:
                    return -1
                return "(" + symbol + ' ' + Y_expand + ' ' + Z_expand + ")"

        tree = expand((0, n - 1), 'ROOT')
        if tree == -1:
            return "FAILED TO PARSE!"
        return tree

    print(sent)
    sent = sent.split(' ')
    n = len(sent)
    Q = get_q(pcfg)
    N = pcfg._rules.keys()
    pi = init(n, N, Q)
    bp = {}

    indices = [ind for gap in xrange(n) for ind in np.ndindex((n, n)) if ind[0] < ind[1] and ind[1] - ind[0] == gap]
    for (i, j) in indices:
        # for X in get_N(i, j):
        for X in N:
            pi_max, pi_arg_max = updatePi(pi, i, j, Q, X)
            pi[i, j, X] = pi_max
            bp[i, j, X] = pi_arg_max
    print 'Pi after: %s' % pi
    print 'BP: %s' % bp
    res = get_derivation(bp)
    if res != "FAILED TO PARSE!":
        return res
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
