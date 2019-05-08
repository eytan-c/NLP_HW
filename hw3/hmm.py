from data import *
import time

from datetime import datetime


def hmm_train(sents):
    """
        sents: list of tagged sentences
        Returns: the q-counts and e-counts of the sentences' tags, total number of tokens in the sentences
    """

    print "Start training"
    total_tokens = 0
    q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = {}, {}, {}, {}, {}
    '''
    used the enumerate(sent) to deal with the cases of first word and second word
    Not sure that the difference between q_uni_counts and e_tag_counts needs to be
    I decided to use the dict.get(key, 0) to check if a key is in the dict and get value if true, and if not return 0
    '''
    ### YOUR CODE HERE
    for sent in sents:
        uni = '*'
        bi = ('*', '*')
        q_uni_counts[uni] = q_uni_counts.get(uni, 0) + 1
        q_bi_counts[bi] = q_bi_counts.get(bi, 0) + 1
        e_tag_counts[uni] = e_tag_counts.get(uni, 0) + 1
        for i, token in enumerate(sent):
            total_tokens += 1
            uni = token[1]
            if i == 0:  # First word of sentence
                tri = ('*', '*', token[1])
                bi = ('*', token[1])
            elif i == 1:  # second word of sentence
                tri = ('*', sent[i - 1][1], token[1])
                bi = (sent[i - 1][1], token[1])
            else:
                tri = (sent[i - 2][1], sent[i - 1][1], token[1])
                bi = (sent[i - 1][1], token[1])

            q_tri_counts[tri] = q_tri_counts.get(tri, 0) + 1
            q_bi_counts[bi] = q_bi_counts.get(bi, 0) + 1
            q_uni_counts[uni] = q_uni_counts.get(uni, 0) + 1

            e_word_tag_counts[token] = e_word_tag_counts.get(token, 0) + 1
            e_tag_counts[uni] = e_tag_counts.get(uni, 0) + 1

        ### Get last bi and tri gram counts
        q_uni_counts['STOP'] = q_uni_counts.get('STOP', 0) + 1
        e_tag_counts['STOP'] = e_tag_counts.get('STOP', 0) + 1
        if len(sent) == 1:
            tri_stop = ('*', sent[len(sent) - 1][1], 'STOP')
            bi_stop = (sent[len(sent) - 1][1], 'STOP')
        else:
            tri_stop = (sent[len(sent) - 2][1], sent[len(sent) - 1][1], 'STOP')
            bi_stop = (sent[len(sent) - 1][1], 'STOP')
        q_tri_counts[tri_stop] = q_tri_counts.get(tri_stop, 0) + 1
        q_bi_counts[bi_stop] = q_bi_counts.get(bi_stop, 0) + 1
    ### END YOUR CODE
    return total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts


def hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts, lambda1,
                lambda2):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Returns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    ### YOUR CODE HERE
    import math

    all_tags = q_uni_counts.keys()
    n = len(sent)
    prun_count = 0
    not_prun_count = 0

    def getSet(set_num):
        if set_num > 0:
            return all_tags
        else:
            return {'*'}

    def calc_prob(v, w, u, model):
        prob = 0.0
        word = v
        prev_word = u
        prev_to_prev_word = w

        if model == "unigram":
            if word in q_uni_counts:
                prob = float(q_uni_counts[word]) / total_tokens

        if model == "bigram":
            if (prev_word, word) in q_bi_counts:
                prob = float(q_bi_counts[(prev_word, word)]) / \
                       q_uni_counts[prev_word]
            else:
                prob = 0
        if model == "trigram":
            if (prev_to_prev_word, prev_word, word) in q_tri_counts:
                prob = float(q_tri_counts[(prev_to_prev_word, prev_word, word)]) \
                       / q_bi_counts[(prev_to_prev_word, prev_word)]
            else:
                prob = 0
        return prob

    def get_q(v, w, u):
        prob = lambda1 * calc_prob(v, w, u, "trigram") + \
               lambda2 * calc_prob(v, w, u, "bigram") + \
               (1 - lambda1 - lambda2) * calc_prob(v, w, u, "unigram")
        return prob
        #if prob == 0:
        #    return -1000
        #return math.log(prob)

    def get_e(x_k, v):
        e_x_k_v = e_word_tag_counts.get((x_k, v), 0)
        if  e_x_k_v == 0:
            return 0
            #return -1000
        else:
            return float(e_x_k_v) / e_tag_counts[v]
            #return math.log(float(e_word_tag_counts.get((x_k, v), 0)) / e_tag_counts[v])

    def updatePi(pi, k, u, v):
        max_pi = float('-inf')
        argmax_pi = None

        e = get_e(sent[k - 1], v)
        if e == 0:  # no such tag for this word in training
            return max_pi, argmax_pi  # max_pi, argmax_pi
        for w in getSet(k - 2):
            # for w in getSet(k - 2):
            current = pi.get((k - 1, w, u)) * get_q(v, w, u) * e
            #current = pi.get((k - 1, w, u),-1000) + get_q(v, w, u) + e
            if current > max_pi:
                max_pi = current
                argmax_pi = w
        return max_pi, argmax_pi

    def get_end_tags(pi):
        max_pi = -1
        best_u = ''
        best_v = ''
        for u in getSet(n - 1):
            for v in getSet(n):
                current_pi = pi[(n, u, v)] * get_q('STOP', u, v)
                #current_pi = pi.get((n, u, v),-1000) + get_q('STOP', u, v)
                if current_pi > max_pi:
                    max_pi = current_pi
                    best_u = u
                    best_v = v
        return best_u, best_v

    def prunning(i):
        DTs = ['the', 'The', 'those', 'Those', 'a', 'an', 'An', 'Another', 'another', 'any', 'Any']
        MDs = ['would', 'can', 'may', 'will']
        VBZs = ['is', 'has', 'does']
        VBs = ['be']

        if sent[i] == ',':
            return ','
        if sent[i] == '.':
            return '.'
        if sent[i] in DTs:
            return 'DT'
        if sent[i] == 'of':
            return 'IN'
        if sent[i] in MDs:
            return 'MD'
        if sent[i] in VBZs:
            return 'VBZ'
        if sent[i] in VBs:
            return 'VB'
        return ''

    pi = {(0, '*', '*'): 1}
    bp = {}

    for k in range(1, n + 1):
        prun = prunning(k - 1)  # x_(k-1) is a word with known tag
        for u in getSet(k - 1):
            for v in getSet(k):
                # max_pi, w = updatePi(pi, k, u, v)
                if prun != '' and v != prun:
                    prun_count += 1
                    max_pi, w = float('-inf'), None
                else:
                    not_prun_count += 1
                    max_pi, w = updatePi(pi, k, u, v)
                pi[(k, u, v)] = max_pi
                bp[(k, u, v)] = w

    predicted_tags[n - 2], predicted_tags[n - 1] = get_end_tags(pi)
    for k in range(n - 2, 0, -1):
        predicted_tags[k - 1] = bp[(k + 2, predicted_tags[k], predicted_tags[k + 1])]
    #print("prun percent:", float(prun_count) / (prun_count + not_prun_count))
    ### END YOUR CODE
    return predicted_tags


def hmm_eval(test_data, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm
    """
    print "Start evaluation"
    acc_viterbi = 0.0
    ### YOUR CODE HERE
    lambda1 = 0.481
    lambda2 = 0.394

    correct = 0
    total = 0

    # (code to find words for pruning)
    # all_pairs = []
    # for sent in test_data:
    #     for pair in sent:
    #         if pair[0]!=',' and pair[0]!='the' and pair[0]!='.':
    #             all_pairs.append(pair)
    #         if pair[0]=='of' and pair[1]!='IN':
    #             print("Not good")
    # most_frequent = max(set(all_pairs), key=all_pairs.count)
    # print(most_frequent)

    for sent in test_data[:100]:
        test_sent = [pair[0] for pair in sent]
        predicted_tags = hmm_viterbi(test_sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                                     e_word_tag_counts,
                                     e_tag_counts, lambda1,
                                     lambda2)
        for i, token in enumerate(sent):
            total += 1
            if predicted_tags[i] == token[1]:
                correct += 1
    acc_viterbi = float(correct) / total
    ### END YOUR CODE
    return acc_viterbi


if __name__ == "__main__":
    start_time = time.time()
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")
    vocab = compute_vocab_count(train_sents)

    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = hmm_train(train_sents)
    acc_viterbi = hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,
                           e_tag_counts)
    print "HMM DEV accuracy: " + str(acc_viterbi)

    train_dev_end_time = time.time()
    print "Train and dev evaluation elapsed: " + str(train_dev_end_time - start_time) + " seconds"

    if os.path.exists("Penn_Treebank/test.gold.conll"):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        acc_viterbi = hmm_eval(test_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                               e_word_tag_counts, e_tag_counts)
        print "HMM TEST accuracy: " + str(acc_viterbi)
        full_flow_end_time = time.time()
        print "Full flow elapsed: " + str(full_flow_end_time - start_time) + " seconds"
