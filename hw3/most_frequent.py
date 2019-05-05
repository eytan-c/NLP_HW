from data import *

def most_frequent_train(train_data):
    """
    Gets training data that includes tagged sentences.
    Returns a dictionary that maps every word in the training set to its most frequent tag.
    """
    ### YOUR CODE HERE
    return get_frequent_tag(train_data)
    ### END YOUR CODE

def get_frequent_tag(data):
    freq_dict = {}
    for sent in data:
        for token in sent:
            if token[0] in freq_dict:
                freq_dict[token[0]][token[1]] = freq_dict[token[0]].get(token[1], 0) + 1
            else:
                freq_dict[token[0]] = {token[1]: 1}
    ret = {}
    for word, tags in freq_dict.items():
        max_tag = max(tags, key=tags.get)
        ret[word] = max_tag
    return ret

def most_frequent_eval(test_set, pred_tags):
    """
    Gets test data and tag prediction map.
    Returns an evaluation of the accuracy of the most frequent tagger.
    """
    ### YOUR CODE HERE
    # test_preds = get_frequent_tag(test_set)
    # acc = 0
    # for word, pred in test_preds.items():
    #     if pred_tags.get(word) == pred: acc += 1
    # print "Correct preds: " + str(acc)
    # return float(acc) / len(test_preds)
    acc = 0
    total = 0
    for sent in test_set:
        for token in sent:
            total += 1
            if pred_tags.get(token[0]) == token[1]:
                acc +=1
    print "Total words: %s" % total
    print "Accuracy count: %s" % acc
    return float(acc) / total
    ### END YOUR CODE

if __name__ == "__main__":
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")
    vocab = compute_vocab_count(train_sents)
    count = 0
    for word, freq in vocab.items():
        if freq < MIN_FREQ : count += 1
    print("Number of words with count < MIN_FREQ:", count)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    model = most_frequent_train(train_sents)
    print "dev: most frequent acc: " + str(most_frequent_eval(dev_sents, model))

    if os.path.exists('Penn_Treebank/test.gold.conll'):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        print "test: most frequent acc: " + str(most_frequent_eval(test_sents, model))