from data import *
import time


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
        for i, token in enumerate(sent):
            total_tokens += 1
            uni = token[1]
            if i == 0: # First word of sentence
                tri = ('*','*',token[1])
                bi = ('*', token[1])
            elif i == 1: # second word of sentence
                tri = ('*', sent[i-1][1], token[1])
                bi = (sent[i-1][1], token[1])
            else:
                tri = (sent[i-2][1], sent[i-1][1], token[1])
                bi = (sent[i-1][1], token[1])
            
            q_tri_counts[tri] = q_tri_counts.get(tri, 0) + 1
            q_bi_counts[bi] = q_bi_counts.get(bi, 0) + 1
            q_uni_counts[uni] = q_uni_counts.get(uni, 0) + 1
            
            e_word_tag_counts[token] = e_word_tag_counts.get(token, 0) + 1
        
        ### Get last bi and tri gram counts
        tri_stop = (sent[len(sent)-2][1], sent[len(sent)-1][1], 'STOP')
        bi_stop = (sent[len(sent)-1][1], 'STOP')
        q_tri_counts[tri_stop] = q_tri_counts.get(tri_stop, 0) + 1
        q_bi_counts[bi_stop] = q_bi_counts.get(bi_stop, 0) + 1
        
    ### END YOUR CODE
    return total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts

def hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts, lambda1, lambda2):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Returns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE
    return predicted_tags

def hmm_eval(test_data, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm
    """
    print "Start evaluation"
    acc_viterbi = 0.0
    ### YOUR CODE HERE
    raise NotImplementedError
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
    acc_viterbi = hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts)
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