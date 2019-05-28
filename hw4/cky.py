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
    def init(n, N, Q):
        pi = {}
        for X in N:
            for i in xrange(n):
                pi[i, i, X] = Q.get((X, (sent[i],)), 0)
        return pi

    def get_q(grammar):
        q_dict = {}
        for left, right in grammar._rules.items():
            # if sent == ['the', 'president', 'thought', 'that', 'a', 'sandwich', 'sighed', '.'] and left == 'S':
            #     print "Break point"
            denom = sum([d[1] for d in right])
            # if denom == 0:
            #     print "Left: %s, Right: %s" % (left, right)
            for der in right:
                q_dict[(left, tuple(der[0]))] = der[1] / float(denom)
        return q_dict

    def updatePi(pi, i, j, Q, X):
        best_pi = float('-inf')
        best_rule = ()
        for rule in Q.keys():
            if rule[0] != X: continue
            if len(rule[1]) > 1:
                if i == 0 and j==1 and X=='NP' and rule[1][0]=='prime':
                    pi1 = pi[(0 + 1, j, rule[1][1])]
                    pi2 = pi[(i, 0, rule[1][0])]
                    q1 = Q.get(rule, 0.0)
                    print("debug")
                for s in xrange(i, j):
                    curr_pi = Q.get(rule, 0.0) * pi[(i, s, rule[1][0])] * pi[(s + 1, j, rule[1][1])]
                    # if sent == ['it', 'perplexed', 'the', 'president', 'that', 'a', 'sandwich', 'ate', 'Sally', '.']:
                    #     if X == 'S' and i==0 and j==8:
                    #         print "break point"
                    #     if X == 'S' and i == 0 and j == 6:
                    #         if rule[1] == ('NP','VerbI'):
                    #             print "break point S"
                    #     elif X == 'NP' and i == 0 and j == 5:
                    #         if rule[1] == ('Det', 'Noun'):
                    #             print "break point NP"
                    #     elif X == 'Noun' and i == 1 and j == 5:
                    #         print "break point Noun"
                    #     elif X == 'Noun' and i == 1 and j == 6:
                    #         print "break point Noun"
                    # if sent == ['it', 'perplexed', 'the', 'president', 'that', 'a', 'sandwich', 'ate', 'Sally', '.']:
                    #     if (X == 'S' and i == 0 and j == 8) or (X=='VP' and i==1 and j==8) or (i==2 and j==8 and X=='NP.SBAR'):
                    #         print "Rule: %s, curr_pi: %s, s: %s" % (rule, curr_pi, s)
                    #     elif X == 'NP' and i == 0 and j == 5:
                    #         print "Rule: %s, curr_pi: %s, s: %s" % (rule, curr_pi, s)
                    #     elif X == 'Noun' and i == 1 and j == 5:
                    #         print "Rule: %s, curr_pi: %s, s: %s" % (rule, curr_pi, s)
                    #     elif X == 'Noun' and i == 1 and j == 6:
                    #         print "Rule: %s, curr_pi: %s, s: %s" % (rule, curr_pi, s)
                    #     if curr_pi == float('nan'):
                    #         print "curr_pi: %s" % curr_pi
                    #         print "rule: %s, (i,j): (%s,%s), s: %s" % (rule,i,j,s)
                    #         print "Q.get: %s, pi[%i,%s,%s]: %s, pi[%s,%s,%s]: %s" % ( Q.get(rule, 0.0),i,s, rule[1][0], pi[(i, s, rule[1][0])], s + 1, j,rule[1][1], pi[(s + 1, j, rule[1][1])])
                    if curr_pi > best_pi:
                        # if sent == ['it', 'perplexed', 'the', 'president', 'that', 'a', 'sandwich', 'ate', 'Sally', '.']:
                        #     if X == 'S' and i == 0 and j == 8:
                        #         print "Pi update!"
                        #     elif X == 'NP' and i == 0 and j == 5:
                        #         print "Pi update!"
                        #     elif X == 'Noun' and i == 1 and j == 5:
                        #         print "Pi update!"
                        #     elif X == 'Noun' and i == 1 and j == 6:
                        #         print "Pi update!"
                        best_pi = curr_pi
                        best_rule = (rule, s)
            elif i == j: # Unary rule
                print("len(rule[1]) <= 1",rule)
                print("i,j", i, j)
        return best_pi, best_rule

    def get_derivation(bp):
        tree = ''

        def expand(index, symbol):
            if bp.get((index[0], index[1], symbol)) is None:
                if pi.get((index[0], index[1], symbol)) > 0:  # non terminal to existing terminal
                    return "(" + symbol + " " + sent[index[0]] + ")"
                else:
                    print("index[0], index[1], symbol",index[0], index[1], symbol)
                    return -1
            # if sent == ['it', 'perplexed', 'the', 'president', 'that', 'a', 'sandwich', 'ate', 'Sally', '.']:
            #     print "break point"
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

        for item in sorted(bp.items()):
            print(item)

        tree = expand((0, n - 1), 'ROOT')
        if tree == -1:
            return "FAILED TO PARSE!"
        return tree

    sent = sent.split(' ')
    n = len(sent)
    Q = get_q(pcfg)
    N = pcfg._rules.keys()
    pi = init(n, N, Q)
    bp = {}

    indices = [ind for gap in xrange(n) for ind in np.ndindex((n, n)) if ind[0] < ind[1] and ind[1] - ind[0] == gap]
    for (i, j) in indices:
        for X in N:
            pi_max, pi_arg_max = updatePi(pi, i, j, Q, X)
            pi[i, j, X] = pi_max
            bp[i, j, X] = pi_arg_max
    res = get_derivation(bp)
    if res != "FAILED TO PARSE!":
        return res
    ### END YOUR CODE
    return "FAILED TO PARSE!"


def non_cnf_cky(pcfg, sent):
    ### YOUR CODE HERE
    def cut_it(rule, j):
        # print("cut_it: rule is-",rule)
        if len(pcfg._rules[rule[0]][j][0]) >= 2:  # more than 2 words in right hand side of rule
            first_word = pcfg._rules[rule[0]][j][0][0]  # first word
            rest_of_words = pcfg._rules[rule[0]][j][0][1:]
            rest_of_words_joints = ''.join([x for x in rest_of_words])
            new_rhs = [first_word, rest_of_words_joints]  # [FIRST_WORD, rest_of_words]
            weight = pcfg._rules[rule[0]][j][1]

            # org_symbol -> [FIRST_WORD, rest_of_words] (non-terminal -> [non-terminal,non-terminal])t)
            pcfg.add_rule(rule[0], new_rhs, weight)
            # delete previous rule's weigth
            pcfg._rules[rule[0]][j] = (pcfg._rules[rule[0]][j][0],0)
            # FIRST_WORD -> [FIRST_WORD] (non-terminal -> [terminal])
            pcfg.add_rule(first_word, [first_word], weight)
            if len(rest_of_words) > 1:
                # rest_of_words_joints - > [rest_of_words] (non-terminal -> list of terminals)
                pcfg.add_rule(rest_of_words_joints, rest_of_words, weight)
                cut_it((rest_of_words_joints, pcfg._rules[rest_of_words_joints]), 0)
            else:
                second_word = rest_of_words[0]
                pcfg.add_rule(second_word, [second_word], weight)

    original_non_terminals = pcfg._rules.keys()
    all_rules = pcfg._rules.items()
    for i, X in enumerate(all_rules):
        for j, rhs in enumerate(X[1]):
            #if len(rhs[0]) >= 2 and:
            if len(rhs[0]) >= 2 and rhs[0][0] not in pcfg._rules and rhs[0][1] not in pcfg._rules : # non-terminal -> list of terminals (len at least 2, not unary)
                # print("all_rules[i]:", all_rules[i])
                # print("all_rules[i][1]:", all_rules[i][1])
                # print("all_rules[i][1][j]:", all_rules[i][1][j])
                # print("all_rules[i][1][j][0]:", all_rules[i][1][j][0])
                # print("pcfg._rules[all_rules[i][0]][j][0]:", pcfg._rules[all_rules[i][0]][j][0])
                # print("")
                cut_it(all_rules[i], j)

    for rule in sorted(pcfg._rules.items()):
        print(rule)

    #return cnf_cky(pcfg, sent)
    #################################### from CNF version ######################################
    import numpy as np
    def init(n, N, Q):
        pi = {}
        for X in N:
            for i in xrange(n):
                pi[i, i, X] = Q.get((X, (sent[i],)), 0)
        return pi

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
            if len(rule[1]) > 1:
                if i == 0 and j==1 and X=='NP' and rule[1][0]=='prime':
                    pi1 = pi[(0 + 1, j, rule[1][1])]
                    pi2 = pi[(i, 0, rule[1][0])]
                    q1 = Q.get(rule, 0.0)
                    print("debug")
                for s in xrange(i, j):
                    curr_pi = Q.get(rule, 0.0) * pi[(i, s, rule[1][0])] * pi[(s + 1, j, rule[1][1])]
                    if curr_pi > best_pi:
                        best_pi = curr_pi
                        best_rule = (rule, s)
            elif i == j: # Unary rule
                print("len(rule[1]) <= 1",rule)
                print("i,j", i, j)
        return best_pi, best_rule

    def get_derivation(bp):
        tree = ''

        def expand(index, symbol):
            if bp.get((index[0], index[1], symbol)) is None:
                if pi.get((index[0], index[1], symbol)) > 0:  # non terminal to existing terminal
                    if symbol in original_non_terminals:
                        return "(" + symbol + " " + sent[index[0]] + ")"
                    else:
                        return sent[index[0]]
                else:
                    print("index[0], index[1], symbol",index[0], index[1], symbol)
                    return -1
            rule, s = bp[index[0], index[1], symbol]
            Y, Z = rule[1]
            if pcfg.is_terminal(symbol):
                return symbol
            else:
                Y_expand = expand((index[0], s), Y)
                Z_expand = expand((s + 1, index[1]), Z)
                if Y_expand == -1 or Z_expand == -1:
                    return -1
                if Y not in original_non_terminals and Z not in original_non_terminals and rule[0] not in original_non_terminals:
                        return Y_expand + ' ' + Z_expand
                else:
                    return "(" + symbol + ' ' + Y_expand + ' ' + Z_expand + ")"

        for item in sorted(bp.items()):
            print(item)

        tree = expand((0, n - 1), 'ROOT')
        if tree == -1:
            return "FAILED TO PARSE!"
        return tree

    sent = sent.split(' ')
    n = len(sent)
    Q = get_q(pcfg)
    N = pcfg._rules.keys()
    pi = init(n, N, Q)
    bp = {}

    indices = [ind for gap in xrange(n) for ind in np.ndindex((n, n)) if ind[0] < ind[1] and ind[1] - ind[0] == gap]
    for (i, j) in indices:
        for X in N:
            pi_max, pi_arg_max = updatePi(pi, i, j, Q, X)
            pi[i, j, X] = pi_max
            bp[i, j, X] = pi_arg_max
    res = get_derivation(bp)
    if res != "FAILED TO PARSE!":
        return res
    ############################################################################################

    ### END YOUR CODE
    return "FAILED TO PARSE!"


if __name__ == '__main__':
    import sys

    cnf_pcfg = PCFG.from_file_assert(sys.argv[1], assert_cnf=True)
    non_cnf_pcfg = PCFG.from_file_assert(sys.argv[2])
    sents_to_parse = load_sents_to_parse(sys.argv[3])
    for sent in sents_to_parse:
        #print cnf_cky(cnf_pcfg, sent)
        print non_cnf_cky(non_cnf_pcfg, sent)
        break
