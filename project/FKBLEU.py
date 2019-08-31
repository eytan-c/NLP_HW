from nltk.translate import bleu_score

import numpy
# import torch
import pyphen


def iBLEU(input_sent, reference, candidate, alpha=0.9):
	"""
	Calculate iBLEU according to Xu et. al. 2016
	:param input_sent: original sentence
	:param reference: the target sentences to test by
	:param candidate: a proposed sentence from the input
	:param alpha: default param 0.9 from Sun and Zhou (2012)
	:return:
	"""
	smooth = bleu_score.SmoothingFunction()
	ref_candidate = bleu_score.sentence_bleu(reference, candidate, smoothing_function=smooth.method7)
	input_candidate = bleu_score.sentence_bleu(input_sent, candidate, smoothing_function=smooth.method7)
	# print('ref_candidate:', '%s; ' % ref_candidate, 'input_candidate:','%s; ' % input_candidate)
	# print('iBLEU (alpha * ref_candidate - (1 - alpha) * input_candidate) =\n %s' % (alpha * ref_candidate - (1 - alpha) * input_candidate))
	return alpha * ref_candidate - (1 - alpha) * input_candidate
	

def FK(text, language='heb'):
	"""
	TODO: count syllables not with heuristic
	Calculate Flesch-Kincaid Index (Kincaid et al 1975) according to Xu et. al. 2016
	:param language: Used for syllables count heuristic
	:param text: Assumes is a list of lists of words (list of sentences as lists of words)
	:return:
	"""
	# if isinstance(text, list) and all(isinstance(sen, list) for sen in text):
	# 	for sen in text:
	# 		assert all(isinstance(w, str) for w in sen)
	
	num_words = 0
	num_sents = 0
	num_syllables = 0
	if language == 'eng':
		parser = pyphen.Pyphen('en_us')
	else:
		parser = None
	# Gather numerical calculations
	for sen in text:
		num_sents += 1
		for word in sen:
			num_words += 1
			if language == 'heb':  # heuristic that each letter is a syllable in hebrew
				num_syllables += len(word)
			elif language == 'eng':  # syllable parser for English
				num_syllables += len(parser.inserted(word).split('-'))
	# print('Words, Sents, Syllables: %s, %s, %s' % (num_words, num_sents, num_syllables))
	return 0.39 * (num_words / num_sents) + 11.8 * (num_syllables / num_words) - 15.59


def FKdiff(input_sent, candidate):
	"""
	
	:param input_sent: Assumes input sent is a list of lists of words
	:param candidate: Assumes candidate is a list of words
	:return:
	"""
	# using torch
	# return torch.nn.functional.sigmoid(FK([candidate]) - FK(input_sent))
	# using python native
	x = FK([candidate]) - FK(input_sent)
	# print('FK(candidate): %s;  FK(input_sent): %s' % (FK([candidate]), FK(input_sent)))
	# print('x: %s' % x)
	# print('FKdiff:', 1 / (1 + numpy.exp(-x)))
	return 1 / (1 + numpy.exp(-x))


def FKBLEU(input_sent, references, candidate):
	"""
	Calculate iBLEU according to Xu et. al. 2016
	:param input_sent: original sentence. Assumes list of lists of words
	:param references: the target sentences to test by. Assumes list of lists of words
	:param candidate: a proposed sentence from the input. List of words.
	:return:
	"""
	# print('Input: %s, refrences: %s, candidate: %s' % (input_sent, references, candidate))
	return iBLEU(input_sent, references, candidate) * FKdiff(input_sent, candidate)


if __name__ == '__main__':
	entries = [['דבר שרת המשפטים ', 'דברי פתיחה של שרת המשפטים ']]
	print('FKBLEU: ', FKBLEU([entries[0][0].strip().split(' ')], [entries[0][1].strip().split(' ')], "הדברים הפתיחה של שרת המשפטים".split(' ')))
