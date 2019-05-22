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
    def init(n, rules):  # need to implement
        pi = {}
        return pi

    def get_N(i, j):
        return {}

    def get_q(rule):
        return 0.0

    def updatePi(pi, i, j, rules):
        return -1, -1

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
    print(pcfg._rules)
    n = len(sent)
    N = get_N(0, n)  # get all tags
    pi = init(n, pcfg._rules)
    bp = {}

    for i in xrange(n - 1):
        for l in xrange(n - i):
            j = i + l
            for X in get_N(i, j):
                pi_max, pi_arg_max = updatePi(pi, i, j, pcfg._rules)
                pi[i, j, X] = pi_max
                bp[i, j, X] = pi_arg_max
    return get_derivation(bp)
    ### END YOUR CODE
    return "FAILED TO PARSE!"


def non_cnf_cky(pcfg, sent):
    ### YOUR CODE HERE
    raise NotImplementedError
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
