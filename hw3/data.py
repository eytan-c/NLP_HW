import os
MIN_FREQ = 4
import re

def invert_dict(d):
    res = {}
    for k, v in d.iteritems():
        res[v] = k
    return res

def read_conll_pos_file(path):
    """
        Takes a path to a file and returns a list of word/tag pairs
    """
    sents = []
    with open(path, "r") as f:
        curr = []
        for line in f:
            line = line.strip()
            if line == "":
                sents.append(curr)
                curr = []
            else:
                tokens = line.strip().split("\t")
                curr.append((tokens[1],tokens[3]))
    return sents

def increment_count(count_dict, key):
    """
        Puts the key in the dictionary if does not exist or adds one if it does.
        Args:
            count_dict: a dictionary mapping a string to an integer
            key: a string
    """
    if key in count_dict:
        count_dict[key] += 1
    else:
        count_dict[key] = 1

def compute_vocab_count(sents):
    """
        Takes a corpus and computes all words and the number of times they appear
    """
    vocab = {}
    for sent in sents:
        for token in sent:
            increment_count(vocab, token[0])
    return vocab

def replace_word(word):
    """
        Replaces rare words with categories (numbers, dates, etc...)
    """
    # categories = ['twoDigitNum','fourDigitNum','containsDigitAndAlpha',
    #               'containsDigitAndDash','containsDigitAndSlash',
    #               'containsDigitAndComma','containsDigitAndPeriod',
    #               'otherNum','allCaps','capPeriod','firstWord',
    #               'initCap','lowerCase','other']
    ### YOUR CODE HERE
    '''
    I decided to use regular expressions because it is simpler to read and think about.
    I tried to make the regexes as simple as possible to match (using the ^ and $ anchors)
    so that it will also be effective.
    The only thing I wasn't able to think of how to do - is the 'firstWord' tag.
    '''
    if re.match(r"^\d{2}$", word):
        return 'twoDigitNum'
    elif re.match(r"^\d{4}$", word):
        return 'fourDigitNum'
    elif re.match(r"\d", word) and re.match(r"[A-Za-z]", word):
        return 'containsDigitAndAlpha'
    elif re.match(r"\d", word) and re.match(r"-", word):
        return 'containsDigitAndDash'
    elif re.match(r"\d", word) and re.match(r"[\\/]", word):
        return 'containsDigitAndSlash'
    elif re.match(r"\d", word) and re.match(r",", word):
        return 'containsDigitAndComma'
    elif re.match(r"\d", word) and re.match(r"\.", word):
        return 'containsDigitAndPeriod'
    elif re.match(r"^\d+$", word):
        return 'otherNum'
    elif re.match(r"^[A-Z]+$", word):
        return 'allCaps'
    elif re.match(r"^[A-Z]\.$", word):
        return 'capPeriod'
    elif re.match(r"^[A-Z][a-z]+\.$", word):
        return 'abbreviation'
    elif re.match(r"^[A-Z][a-z]+$", word):
        return 'initCap'
    elif word.lower() == word:
        return 'lowerCase'
    elif re.match(r"^[?!@#$%^&*()\[\]{}\-+=;:'\"/\\,.~`]$", word):
        return 'punctuation'
    ### END YOUR CODE
    return "UNK"

def preprocess_sent(vocab, sents):
    """
        return a sentence, where every word that is not frequent enough is replaced
    """
    res = []
    total, replaced = 0, 0
    for sent in sents:
        new_sent = []
        for token in sent:
            if token[0] in vocab and vocab[token[0]] >= MIN_FREQ:
                new_sent.append(token)
            else:
                new_sent.append((replace_word(token[0]), token[1]))
                replaced += 1
            total += 1
        res.append(new_sent)
    print "num replaced: " + str(replaced)
    print "total: " + str(total)
    print "replaced: " + str(float(replaced)/total)
    return res







