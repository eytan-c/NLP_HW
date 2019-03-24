#!/usr/bin/env python

import numpy as np
import random

from q1b_softmax import softmax
from q1e_gradcheck import gradcheck_naive
from q1d_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """
    x = x / np.sqrt(np.sum(x**2,keepdims=True,axis=1))
    return x


def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    print x
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print ""


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    y_hat = softmax(np.dot(outputVectors,predicted))
    y = np.zeros(outputVectors.shape[0])
    y[target] = 1

    cost =  np.log(y_hat[target]) #CE is log of predict probebilty according to 1-hot vector
    gradPred = np.dot(outputVectors.transpose(), (y_hat - y))# U[y^hat - y]

    temp = y_hat - y
    grad = np.multiply(np.expand_dims(temp, 1),predicted)  # (y_w^hat - y_w)v_c

    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    ### YOUR CODE HERE
    """
    Arguments:
    predicted -- v_c
    target -- o in the notations
    outputVectors -- all the U's (but as rows and not as columns (need to transpose)
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- neg-sampling cost
    gradPred -- dJ/dv_c
    grad -- dJ/dU
    """
    target_pred_dot_sig = sigmoid(np.dot(outputVectors[indices[0]], predicted)) # s(u_o^T * v_c)
    sample_pred_dot_sig = sigmoid(-np.dot(outputVectors[indices[1:]], predicted)) # s(u_k^T * v_c) as whole matrix
    log_part = np.log(target_pred_dot_sig)
    sum_part = np.sum(np.log(sample_pred_dot_sig))
    cost = - log_part - sum_part
    
    ## (s(U*v_c) -[1,00]) * v_c
    # e_1 = np.zeros(len(indices))
    # e_1[target] = 1
    # grad_calc = (sigmoid(np.dot(outputVectors[indices], predicted)) - e_1).reshape(-1, 1) * predicted
    # grad = np.zeros_like(outputVectors)
    # grad[target] = (sigmoid(outputVectors[]))
    # for i in xrange(len(indices)):
    #     grad[indices[i]] = grad_calc[i].copy()
    probs = outputVectors.dot(predicted)
    grad = np.zeros_like(outputVectors)
    grad[target] = (sigmoid(probs[target]) - 1) * predicted

    for k in indices[1:]:
      grad[k] += (1.0 - sigmoid(-np.dot(outputVectors[k], predicted))) * predicted
    
    ## -(1-s(u_o^T * v_c)) * u_o^T + sum_K[(1-s(-u_k^T * v_c)) * u_k^T]
    gradPred = -1 * (1 - target_pred_dot_sig) * outputVectors[indices[0]] \
               + np.sum((1 - sample_pred_dot_sig).reshape(-1,1) * outputVectors[indices[1:]], axis=0)
    ### END YOUR CODE

    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    centerWordIndex = tokens[currentWord]
    v_c = inputVectors[centerWordIndex]

    for contextCurrentWord in contextWords:
        u_o_index = tokens[contextCurrentWord]
        currentCost, currentGradIn, currentGradOut = word2vecCostAndGradient(v_c, u_o_index, outputVectors, dataset)
        cost += currentCost
        gradIn[centerWordIndex] += currentGradIn
        gradOut += currentGradOut

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    # gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
    #     skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
    #     dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)

    print "\n=== Results ==="
    # print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
    #     dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
