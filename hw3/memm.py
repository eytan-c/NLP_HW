from data import *
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
import time
import numpy as np
import pickle


def build_extra_decoding_arguments(train_sents):
    """
    Receives: all sentences from training set
    Returns: all extra arguments which your decoding procedures requires
    """

    extra_decoding_arguments = {}
    ### YOUR CODE HERE
    total_tokens = 0
    q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = {}, {}, {}, {}, {}
    for sent in train_sents:
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
    
    extra_decoding_arguments = {"total_tokens": total_tokens,
                                "q_tri_counts": q_tri_counts,
                                "q_bi_counts": q_bi_counts,
                                "q_uni_counts": q_uni_counts,
                                "e_word_tag_counts": e_word_tag_counts,
                                "e_tag_counts": e_tag_counts}
    ### END YOUR CODE

    return extra_decoding_arguments


def extract_features_base(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag):
    """
        Receives: a word's local information
        Returns: The word's features.
    """
    features = {}
    features['word'] = curr_word
    ### YOUR CODE HERE
    features['prev_tag'] = prev_tag
    features['prevprev_tag'] = prevprev_tag
    features['prev_tag_bigram'] = prev_tag + ' ' + prevprev_tag
    features['prev_word'] = prev_word
    features['prevprev_word'] = prevprev_word
    features['next_word'] = next_word
    # features['prev_word-tag'] = (prev_word, prev_tag)
    # features['prevprev_word-tag'] = (prevprev_word, prevprev_tag)
    features['initCap'] = 1 if re.match(r"^[A-Z]", curr_word[0]) else 0
    features['hasNumeric'] = 1 if re.search(r"\d", curr_word[0]) else 0
    features['hasDash'] = 1 if re.search(r"-", curr_word[0]) else 0
    features['allCaps'] = 1 if re.search(r"^[A-Z]+$", curr_word[0]) else 0
    #### suffixes + prefixes
    if len(curr_word) > 5:
        for i in xrange(4):
            features['prefix' + str(i + 1)] = curr_word[:i + 1]
            features['suffix' + str(i + 1)] = curr_word[- (i + 1):]
    ### END YOUR CODE
    return features

def extract_features(sentence, i):
    curr_word = sentence[i][0]
    prev_token = sentence[i - 1] if i > 0 else ('<st>', '*')
    prevprev_token = sentence[i - 2] if i > 1 else ('<st>', '*')
    next_token = sentence[i + 1] if i < (len(sentence) - 1) else ('</s>', 'STOP')
    return extract_features_base(curr_word, next_token[0], prev_token[0], prevprev_token[0], prev_token[1], prevprev_token[1])

def vectorize_features(vec, features):
    """
        Receives: feature dictionary
        Returns: feature vector

        Note: use this function only if you chose to use the sklearn solver!
        This function prepares the feature vector for the sklearn solver,
        use it for tags prediction.
    """
    example = [features]
    return vec.transform(example)

def create_examples(sents, tag_to_idx_dict):
    examples = []
    labels = []
    num_of_sents = 0
    for sent in sents:
        num_of_sents += 1
        for i in xrange(len(sent)):
            features = extract_features(sent, i)
            examples.append(features)
            labels.append(tag_to_idx_dict[sent[i][1]])

    return examples, labels


def memm_greedy(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    """
    logreg - class sklearn.linear_model.LinearRegression
    vec - class sklearn.feature_extraction.DictVectorizer
    index_to_tag_dict - class dictionary from number to
    extra_decoding_arguments
    """
    ### YOUR CODE HERE
    for j, word in enumerate(sent):
        features = extract_features(sent, j)
        word_vec = vec.transform(features)
        tag = int(logreg.predict(word_vec))
        # prob = logreg.predict_proba(word_vec)
        predicted_tags[j] = index_to_tag_dict[tag]  # tag is ndarray
        
    ### END YOUR CODE
    return predicted_tags

def memm_viterbi(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    
    # """
    # ###
    # # Reference
    # ###
    #
    # predicted_tags = [""] * (len(sent))
    #
    # ### YOUR CODE HERE
    # e_word_tag_counts = extra_decoding_arguments['e_word_tag_counts']
    # tagset = tag_to_idx_dict
    # _cache = {}
    # num_tags = len(tagset)
    # rng = np.arange(num_tags)
    # pi0 = -np.inf * np.ones((num_tags, num_tags))
    # pi0[tagset['*'], tagset['*']] = 0
    # pi = [pi0]
    # bp = []
    # for k, x_k in enumerate(sent):
    #     pi_k = -np.inf * np.ones((num_tags, num_tags))
    #     bp_k = np.zeros((num_tags, num_tags), int)
    #     curr_word = x_k
    #     for u in xrange(num_tags - 1) if k > 0 else [tagset['*']]:
    #         tag_u = index_to_tag_dict[u]
    #         prev_token = (sent[k - 1], tag_u) if k > 0 else ('<s>', '*')
    #         if k > 0 and e_word_tag_counts.get(prev_token, 0) == 0:
    #             continue
    #         next_token = sent[k + 1] if k < (len(sent) - 1) else ('</s>', 'STOP')
    #         log_q = np.zeros((num_tags, num_tags))
    #         for w in xrange(num_tags - 1) if k > 1 else [tagset['*']]:
    #             tag_w = index_to_tag_dict[w]
    #             prevprev_token = (sent[k - 2], tag_w) if k > 1 else ('<s>', '*')
    #             prob = _cache.get(
    #                     (curr_word, next_token, prev_token[0], prevprev_token[0], prev_token[1], prevprev_token[1]))
    #             if prob is None:
    #                 features = extract_features_base(curr_word, next_token, prev_token[0], prevprev_token[0],
    #                                                  prev_token[1], prevprev_token[1])
    #                 vectorized_sent = vec.transform(features)
    #                 _cache[curr_word, next_token, prev_token[0], prevprev_token[0], prev_token[1], prevprev_token[
    #                     1]] = prob = logreg.predict_log_proba(vectorized_sent)[0]
    #             log_q[w, 0:num_tags-1] = prob  ##TODO the problem is that the indexing is bad
    #
    #         bp_k[u, :] = w = np.argmax(pi[-1][:, u, None] + log_q, axis=0)
    #         pi_k[u, :] = pi[-1][w, u] + log_q[w, rng]
    #
    #     pi.append(pi_k)
    #     bp.append(bp_k)
    #
    # yn1, yn = np.unravel_index(np.argmax(pi[-1]), pi[-1].shape)
    # predicted_tags[-1] = index_to_tag_dict[yn]
    #
    # if len(sent) == 1:
    #     return predicted_tags
    #
    # predicted_tags[-2] = index_to_tag_dict[yn1]
    #
    # for k in range(len(sent) - 3, -1, -1):
    #     tag1 = predicted_tags[k + 1]
    #     tag2 = predicted_tags[k + 2]
    #     yk = bp[k + 2][tagset[tag1], tagset[tag2]]
    #     predicted_tags[k] = index_to_tag_dict[yk]
    # """

    
    predicted_tags = [""] * (len(sent))
    ### YOUR CODE HERE
    ## working with log probabilities for stability

    n = len(sent)

    def getSet(set_num, S):
        '''
				gets all possible tags for word in index set_num - 1
				'''
        if set_num > 0:
            word = sent[set_num - 1][0]
            if word not in S:
                prun = prunning(set_num - 1)
                if prun != '':  # if word has pre known tag
                    S[word] = [prun]
                else:  # the only possible tags are one seen in the train for this word
                    S[word] = [tag for (w, tag) in extra_decoding_arguments["e_word_tag_counts"] if w == word]
            return S[word]
        else:
            return ['*']

    def prunning(i):
        '''

        :param i: word index in sent to check if tag is known
        :return: if words is one in known words lists get back it's tag, return empty '' o.w
        '''
        DTs = {'the', 'The', 'those', 'Those', 'a', 'an', 'An', 'Another', 'another', 'any', 'Any'}
        MDs = {'would', 'can', 'may', 'will'}
        VBZs = {'is', 'has', 'does'}
        VBs = {'be'}

        word = sent[i]
        if word == ',':
            return ','
        if word == '.':
            return '.'
        if word in DTs:
            return 'DT'
        if word == 'of':
            return 'IN'
        if word in MDs:
            return 'MD'
        if word in VBZs:
            return 'VBZ'
        if word in VBs:
            return 'VB'
        return ''
    
    def calculatePi(pi, k, u, curr_word, sent, num_tags, tag_to_index_dict=tag_to_idx_dict, e_word_tag_counts=extra_decoding_arguments['e_word_tag_counts'], logreg=logreg, vec=vec):
        Q = -np.inf * np.ones((num_tags, num_tags - 1))
        if (sent[k - 2], u) not in e_word_tag_counts: 0  ##TODO This never happens because the makeSet take care of this. Need to make sure (itay) - you are right, delete this
        for t in getSet(k - 2, S):  # a list of valid tags as strings
            t_index = tag_to_index_dict[t]
            ##### Calculate q(v|t,u,w,k) for a specific t, and specific u
            next_token = sent[k] if k < n else ("</s>", "STOP")
            prev_word = sent[k - 2] if k > 1 else "<st>"
            prevprev_word = sent[k - 3] if k > 2 else "<st>"
            word_features = extract_features_base(curr_word, next_token[0], prev_word, prevprev_word, u, t)
            word_vec = vectorize_features(vec, word_features)
            Q[t_index, :] = logreg.predict_log_proba(word_vec)  ## for every u we have such a matrix
            ##### END Calculate q(v|t,u,w,k)
            Q[t_index, :] = pi.get((k - 1, t, u), 0) + Q[t_index,:]  ## the matrix Q is per each u, and for each t we need to add the probabilities from before if exist. If they don't then -inf ## TODO might need to change the defualt falue of the pi.get() to -np.inf
        ### And the end of the loop on t, we have a matrix Q with rows indexed by tag t, and columns indexed by tag v. Thus, we need to calculate the max and argmax of each columns, and they are the correponding values for the pi(k,u,v)
        max_probs = np.max(Q, axis=0)  # axis=0 is columns
        argmax_probs = np.argmax(Q, axis=0)
        return max_probs, argmax_probs

    def get_end_tags(pi):
        max_pi = -np.inf
        best_u = ''
        best_v = ''

        for u in getSet(n - 1, S):
            for v in getSet(n, S):
                current_pi = pi[(n, u, v)]
                if current_pi > max_pi:
                    max_pi = current_pi
                    best_u = u
                    best_v = v
        return best_u, best_v
    
    pi = {(0, '*', '*'): 1}
    bp = {}
    S = {}
    for k in xrange(1, n + 1):  # notice k is corrsponded with index k-1 in sent
        curr_word = sent[k - 1][0]  # sen is (word, tag) pairs
        for u in getSet(k - 1, S):
            max_probs, argmax_probs = calculatePi(pi, k, u, curr_word, sent, num_tags=len(index_to_tag_dict))  # needs to return the max and argmax by t of pi(k-1,t,u) * q(v|(t,u,w,k)), t in S_k-2
            for i in range(len(max_probs)):  # for v in S_k
                v = index_to_tag_dict[i]
                if max_probs[i] != -np.inf:  # makes sure that this is valid
                    pi[(k, u, v)] = max_probs[i]
                    bp[(k, u, v)] = index_to_tag_dict[argmax_probs[i]]
        
    
    predicted_tags[n - 2], predicted_tags[n - 1] = get_end_tags(pi)
    for k in xrange(n - 2, 0, -1):
        predicted_tags[k - 1] = bp[(k + 2, predicted_tags[k], predicted_tags[k + 1])]
    ### END YOUR CODE
    return predicted_tags
    
    
    # ##########
    # ### My Initial Try
    # ##########
    #
    # def getSet(word, index, index_to_tag_dict):
    #  if index >= 0:
    #         S = {(word[0], tag, idx) for idx, tag in index_to_tag_dict.items() \
    #              if extra_decoding_arguments['e_word_tag_counts'].get((word[0], tag),0) > 0}
    #     else:
    #         S = {('<st>', '*', len(index_to_tag_dict) - 1)}
    #     return S
    #
    # tag_rng = np.array(sorted(index_to_tag_dict.keys()))
    # num_tags = len(tag_rng)
    # pi0 = -np.inf * np.ones((num_tags, num_tags))
    # pi0[tag_rng[-1],tag_rng[-1]] = 0  # pi(0,*,*) = log(1)
    # pi = [pi0]
    # bp = []
    # _cache = {}
    # for k in xrange(len(sent)):
    #     pi_k = -np.inf * np.ones((num_tags, num_tags))
    #     bp_k = np.zeros((num_tags, num_tags), int)
    #     curr_token = sent[k]
    #     next_token = sent[k + 1] if k < len(sent) - 1 else ('</s>', 'STOP')
    #     S_k = getSet(curr_token, k, index_to_tag_dict)
    #     S_k_1 = getSet(curr_token, k - 1, index_to_tag_dict)
    #     S_k_2 = getSet(curr_token, k - 2, index_to_tag_dict)
    #     ## Calculate log-probabilities
    #     log_q = np.zeros((num_tags,num_tags))
    #     for u in S_k_1:
    #         for t in S_k_2:
    #             q = _cache.get((curr_token, next_token[0], u[0], t[0], u[1], t[1]))
    #             if q is None:
    #                 features = extract_features_base(curr_token, next_token[0], u[0], t[0], u[1], t[1])
    #                 feat_vec = vec.transform(features)
    #                 # The next two lines are to fix the mismatch of not seeing '*' label in training
    #                 probs = logreg.predict_log_proba(feat_vec)[0]
    #                 _cache[(curr_token, next_token[0], u[0], t[0], u[1], t[1])] = q = np.append(probs, -np.inf*np.ones(1))
    #             log_q[t[2],:] = q
    #         bp_k[u[2], :] = w = np.argmax(pi[-1][:, u[2], None] + log_q, axis=0)
    #         pi_k[u[2], :] = pi[-1][w, u[2]] + log_q[w, tag_rng]
    #     pi.append(pi_k)
    #     bp.append(bp_k)
    #
    # yn1, yn = np.unravel_index(np.argmax(pi[-1]), pi[-1].shape)
    # predicted_tags[-1] = yn
    #
    # if len(sent) == 1:
    #     predicted_tags = [index_to_tag_dict[label] for label in predicted_tags]
    #     return predicted_tags
    #
    # predicted_tags[-2] = yn1
    # for k in range(len(sent) - 3, -1, -1):
    #     tag1 = predicted_tags[k + 1]
    #     tag2 = predicted_tags[k + 2]
    #     yk = bp[k + 2][tag1, tag2]
    #     predicted_tags[k] = yk
    #
    # for j, label in enumerate(predicted_tags):
    #     predicted_tags[j] = index_to_tag_dict[label]
    # ### END YOUR CODE
    # return predicted_tags

def should_log(sentence_index):
    if sentence_index > 0 and sentence_index % 10 == 0:
        if sentence_index < 150 or sentence_index % 200 == 0:
            return True

    return False


def memm_eval(test_data, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
    Receives: test data set and the parameters learned by memm
    Returns an evaluation of the accuracy of Viterbi & greedy memm
    """
    acc_viterbi, acc_greedy = 0.0, 0.0
    eval_start_timer = time.time()

    correct_greedy_preds = 0
    correct_viterbi_preds = 0
    total_words_count = 0

    for i, sen in enumerate(test_data):
        ### YOUR CODE HERE
        ### Make sure to update Viterbi and greedy accuracy
        # sen_features = [extract_features(sen, i) for i in xrange(len(sen))]
        # sen_vec = vec.transform(sen_features)
        total_words_count += len(sen)
        greedy_tags = memm_greedy(sen, logreg, vec, index_to_tag_dict, extra_decoding_arguments)
        greedy_num = sum([greedy_tags[k] == sen[k][1] for k in xrange(len(sen))])
        correct_greedy_preds += greedy_num
        viterbi_tags = memm_viterbi(sen, logreg, vec, index_to_tag_dict, extra_decoding_arguments)
        viterbi_num = sum([viterbi_tags[k] == sen[k][1] for k in xrange(len(sen))])
        correct_viterbi_preds += viterbi_num
        acc_greedy = float(greedy_num) / len(sen)
        acc_viterbi = float(viterbi_num) / len(sen)
        ### END YOUR CODE

        if should_log(i):
            if acc_greedy == 0 and acc_viterbi == 0:
                raise NotImplementedError
            eval_end_timer = time.time()
            print str.format("Sentence index: {} greedy_acc: {}    Viterbi_acc:{} , elapsed: {} ", str(i), str(acc_greedy), str(acc_viterbi) , str (eval_end_timer - eval_start_timer))
            eval_start_timer = time.time()

    acc_greedy = float(correct_greedy_preds) / float(total_words_count)
    acc_viterbi = float(correct_viterbi_preds) / float(total_words_count)

    return str(acc_viterbi), str(acc_greedy)

def build_tag_to_idx_dict(train_sentences):
    curr_tag_index = 0
    tag_to_idx_dict = {}
    for train_sent in train_sentences:
        for token in train_sent:
            tag = token[1]
            if tag not in tag_to_idx_dict:
                tag_to_idx_dict[tag] = curr_tag_index
                curr_tag_index += 1

    tag_to_idx_dict['*'] = curr_tag_index
    return tag_to_idx_dict


if __name__ == "__main__":
    full_flow_start = time.time()
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")

    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    if os.path.exists("pickles\\args.pkl"):
        print "Opening arg.pkl...."
        with open("C:\\School\\nlp\\NLP_HW\\hw3\\pickles\\args.pkl", 'rb') as f:
            extra_decoding_arguments = pickle.load(f)
    else:
        extra_decoding_arguments = build_extra_decoding_arguments(train_sents)
        with open("C:\\School\\nlp\\NLP_HW\\hw3\\pickles\\args.pkl", 'wb') as f:
            pickle.dump(extra_decoding_arguments, f, protocol=-1)
    dev_sents = preprocess_sent(vocab, dev_sents)
    tag_to_idx_dict = build_tag_to_idx_dict(train_sents)
    index_to_tag_dict = invert_dict(tag_to_idx_dict)

    vec = DictVectorizer()
    print "Create train examples"
    train_examples, train_labels = create_examples(train_sents, tag_to_idx_dict)


    num_train_examples = len(train_examples)
    print "#example: " + str(num_train_examples)
    print "Done"

    print "Create dev examples"
    dev_examples, dev_labels = create_examples(dev_sents, tag_to_idx_dict)
    num_dev_examples = len(dev_examples)
    print "#example: " + str(num_dev_examples)
    print "Done"

    all_examples = train_examples
    all_examples.extend(dev_examples)

    end_make_vars = time.time()
    
    print "Making var took %s" % (full_flow_start - end_make_vars)
    #### For faster debugging - saved all the non Q4 objects in pickle###
    if os.path.exists("C:\\Users\\eytanc\\Documents\\GitHub\\NLP_HW\\NLP_HW\\hw3\\pickles\\model.pkl"):
        print "Opening model.pkl...."
        with open("C:\\Users\\eytanc\\Documents\\GitHub\\NLP_HW\\NLP_HW\\hw3\\pickles\\model.pkl", 'rb') as f:
            logreg, vec = pickle.load(f)
    else:
        print "Vectorize examples"
        all_examples_vectorized = vec.fit_transform(all_examples)
        train_examples_vectorized = all_examples_vectorized[:num_train_examples]
        dev_examples_vectorized = all_examples_vectorized[num_train_examples:]
        print "Done"
        # t_l_set = set(train_labels)
        logreg = linear_model.LogisticRegression(
            multi_class='multinomial', max_iter=128, solver='lbfgs', C=100000, verbose=1)
        print "Fitting..."
        start = time.time()
        logreg.fit(train_examples_vectorized, train_labels)
        end = time.time()
        print "End training, elapsed " + str(end - start) + " seconds"
        # End of log linear model training

        # with open("C:\\Users\\eytanc\\Documents\\GitHub\\NLP_HW\\NLP_HW\\hw3\\pickles\\initial_objs.pkl", 'wb') as f:
        #     pickle.dump([train_sents, dev_sents,vocab,extra_decoding_arguments,tag_to_idx_dict,index_to_tag_dict], f, protocol=-1)
        # with open("C:\\Users\\eytanc\\Documents\\GitHub\\NLP_HW\\NLP_HW\\hw3\\pickles\\data_objs.pkl", 'wb') as f:
        #     pickle.dump([train_examples,train_labels,dev_examples,dev_labels,all_examples_vectorized,train_examples_vectorized,dev_examples_vectorized], f, protocol=-1)
        with open("C:\\Users\\eytanc\\Documents\\GitHub\\NLP_HW\\NLP_HW\\hw3\\pickles\\model_2.pkl", 'wb') as f:
            pickle.dump([logreg, vec], f, protocol=-1)
    
    
    # Evaluation code - do not make any changes
    start = time.time()
    print "Start evaluation on dev set"

    acc_viterbi, acc_greedy = memm_eval(dev_sents, logreg, vec, index_to_tag_dict, extra_decoding_arguments)
    end = time.time()
    print "Dev: Accuracy greedy memm : " + acc_greedy
    print "Dev: Accuracy Viterbi memm : " + acc_viterbi

    print "Evaluation on dev set elapsed: " + str(end - start) + " seconds"
    if os.path.exists('Penn_Treebank/test.gold.conll'):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        start = time.time()
        print "Start evaluation on test set"
        acc_viterbi, acc_greedy = memm_eval(test_sents, logreg, vec, index_to_tag_dict, extra_decoding_arguments)
        end = time.time()

        print "Test: Accuracy greedy memm: " + acc_greedy
        print "Test:  Accuracy Viterbi memm: " + acc_viterbi

        print "Evaluation on test set elapsed: " + str(end - start) + " seconds"
        full_flow_end = time.time()
        print "The execution of the full flow elapsed: " + str(full_flow_end - full_flow_start) + " seconds"