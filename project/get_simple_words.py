import re
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import numpy
import json

SOS_TOKEN = 0
EOS_TOKEN = 1
UNKNOWN = 2
MAX_LENGTH = 30

train_LM = False
train_LM_data_size = 101#00 # 6000 is everything
data_size = 101 # 6000 is everything

data_dir = 'C:\\Users\\eytanc\\OneDrive\\Documents\\University\\2018-9\\Sem B\\NLP\\Project\\Dataset\\'
data_path = data_dir + 'sim_dataset_23082019.csv'
# lm_data_path = '/content/drive/My Drive/Colab Notebooks/nlp/data/he_htb-ud-train.txt'


class Vocab:
	def __init__(self, name):
		self.name = name
		self.word2index = {}
		self.index2word = {UNKNOWN: '__unk__'}
		self.n_words = 3
	
	def index_sentence(self, sentence, write=True):
		indexes = []
		for w in sentence.strip().split(' '):
			indexes.append(self.index_word(w, write))
		return indexes
	
	def index_word(self, word, write=True):
		if word not in self.word2index:
			if write:
				self.word2index[word] = self.n_words
				self.index2word[self.n_words] = word
				self.n_words = self.n_words + 1
			else:
				return UNKNOWN
		return self.word2index[word]


"""#### Read and organize sentances"""


def normalize_string(s, only_heb=False):
	s = s.lower().strip()
	s = re.sub(r"([.!?])", r" \1", s)
	s = re.sub(r"[\u0590-\u05CF]+", "", s)
	if only_heb:
		s = re.sub(r"[^א-ת.!?\"]+", r" ", s)
		s = re.sub(r"(^|\s)\"(\w)", r"\1\2", re.sub(r"(\w)\"(\s|$)", r"\1\2", s))
	else:
		s = re.sub(r"[^a-zA-Zא-ת.!?\"]+", r" ", s)
		s = re.sub(r"(^|\s)\"(\w)", r"\1\2", re.sub(r"(\w)\"(\s|$)", r"\1\2", s))
	return s


def read_langs(lang1, lang2):
	print("Reading lines...")
	
	if 'normal_heb' == lang1 or 'simple_heb' == lang2:
		df = pd.read_csv(data_path, error_bad_lines=False)  # , encoding='utf-8')
		all_reg_sent = df['reg_sent']
		all_sim_sent = df['sim_sent']
		
		pairs = []
		for i in range(len(all_reg_sent)):
			pairs.append([normalize_string(all_reg_sent[i], only_heb=True), normalize_string(all_sim_sent[i], only_heb=True)])
	
	if 'simple-wiki' == lang1 and 'normal-wiki' == lang2:
		# Read the file and split into lines
		lines = open('/content/drive/My Drive/Colab Notebooks/nlp/data/%s-%s.txt' % (lang1, lang2)).read().strip().split(
			'\n')
		
		# Split every line into pairs and normalize
		lines = [line.split('\t')[2] for line in lines]
		pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
		
		pairs = [pairs[i] + pairs[i + 1] for i in range(0, len(lines), 2)]
	
	# if 'lm_train_heb' == lang1 and 'lm_train_heb' == lang2:
	# 	lines = open(lm_data_path).read().strip().replace('\n', ' ').split('. ')
	# 	pairs = [[normalize_string(line, only_heb=True), normalize_string(line, only_heb=True)] for line in lines]
	
	return pairs


def print_pair(p):
	print(p[0])
	print(p[1])


def filter_pair(p):
	return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH


def filter_pairs(pairs):
	return [pair for pair in pairs if filter_pair(pair)]


def prepare_data(lang1_name, lang2_name):
	pairs = read_langs(lang1_name, lang2_name)
	print("Read %s sentence pairs" % len(pairs))
	
	pairs = filter_pairs(pairs)
	
	print("Trimmed to %s sentence pairs" % len(pairs))
	
	#     print("Indexing words...")
	#     for pair in pairs:
	#         input_vocab.index_sentence(pair[0])
	#         output_vocab.index_sentence(pair[1])
	return pairs


def init_data(input_vocab, output_vocab, dic_reg_to_sim):
	# input_vocab, output_vocab, pairs = prepare_data('simple-wiki', 'normal-wiki', True)
	
	if train_LM:
		pairs = prepare_data('lm_train_heb', 'lm_train_heb')
		# build_dic_reg_to_sim()
		X_lm = []
		y_lm = []
		
		print("Indexing LM train data...")
		
		for pair in pairs:
			if pair[0] != ' ' and pair[1] != ' ':  # delete empty pairs
				input_vocab.index_sentence(pair[0])
				output_vocab.index_sentence(pair[1])
				X_lm.append(pair[0])
				y_lm.append(pair[1])
		X_lm = numpy.asarray(X_lm[:train_LM_data_size])
		y_lm = numpy.asarray(y_lm[:train_LM_data_size])
		
		print("Trimmed LM to %s non-empty sentence pairs" % len(X_lm))
		
		# Print example
		i = random.randint(0, len(X_lm) - 1)
		print(X_lm[i])
		print(y_lm[i])
	
	pairs = prepare_data('normal_heb', 'simple_heb')
	
	print("Indexing data...")
	
	X = []
	y = []
	
	for pair in pairs:
		if pair[0] != ' ' and pair[1] != ' ':  # delete empty pairs
			input_vocab.index_sentence(pair[0])
			output_vocab.index_sentence(pair[1])
			X.append(pair[0])
			y.append(pair[1])
	
	# dic_reg_to_sim = build_dic_reg_to_sim()
	dic_reg_to_sim = {}
	print("Trimmed to %s non-empty sentence pairs" % len(X))
	
	# Print example
	i = random.randint(0, len(X) - 1)
	print(X[i])
	print(y[i])
	
	X_train, X_test, y_train, y_test = train_test_split(numpy.asarray(X[:data_size]), numpy.asarray(y[:data_size]),
	                                                    test_size=0.1, random_state=1)
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1)
	
	print("Train size:", len(X_train))
	print("Validation size:", len(X_val))
	print("Test size:", len(X_test))
	
	print("Data is ready!")
	if train_LM:
		return input_vocab, output_vocab, dic_reg_to_sim, X_train, y_train, X_val, y_val, X_test, y_test, X_lm, y_lm
	else:
		return input_vocab, output_vocab, dic_reg_to_sim, X_train, y_train, X_val, y_val, X_test, y_test


input_vocab = Vocab("Heb-reg")
output_vocab = Vocab("Heb-simple")
dic_reg_to_sim = {}

# json_dir = 'lexicon.json'

# with open(json_dir,'r',encoding='utf-8') as f:
# 	dic_reg_to_sim = json.load(f)


def simplify(word, lexicon):
	"""
	Assumes that the word is in the lexicon.
	Uses random.choice() since, if there are multiple simplifications, we want to
	pick randomly for them. If the word was simplified in the same way in
	multiple entries, then the .choice() accounts for the number of time they
	occur.
	:param str word: word to simplify
	:param dict lexicon: dict to map word to simple version
	:return:
	"""
	sim = random.choice(lexicon[word])[1]
	if type(sim) == str:
		sim = [sim]
	return " ".join(sim)


def get_simple_word_dict(org_sent, target_sent, lexicon, multi_word=0):
	"""
	Deal with no word in sentence: Pick random shared word. If non exists, pick random.
	Deal with multiple words in sentence: Pick first or Pick random.
																				Random can be weighted or not.
																				Default first (multi_word=0)
	If multiple simplification - Random by weight - do be dealt with simplify() function
	Need index in target sentence in Train. If simplification not in there, then pick random.
	:param org_sent:
	:param target_sent:
	:param lexicon:
	:param multi_word:
	:return:
	"""
	potential = []
	org_sent_split = org_sent.split(' ')[:-1]
	target_sent_split = target_sent.split(' ')[:-1]
	for i, word in enumerate(org_sent_split):
		if word in lexicon.keys():
			potential.append((i, word))
	if len(potential) > 1:  # Multiple Potential Words to Simplify
		if multi_word == 0:  # Heuristic - Pick first word in sentence to simplify
			simplified = simplify(potential[0][1], lexicon)  # returns a string (can be multiple words)
			reg_index = potential[0][0]
		else:  # Heuristic pick randomly by number of times simplified
			# Option 1 - Random choice with ints
			# rand_ind = random.randint(0, len(potential) - 1)
			# simplified = simplify(potential[rand_ind][1], lexicon)  # returns a string (can be multiple words)
			# reg_index = potential[rand_ind][0]
			# Option 2 - Random choice with choice
			# choice = random.choice(potential)
			# simplified = simplify(choice[1], lexicon)
			# reg_index = choice[0]
			# Option 3 - Random choice with choices (weighted
			ws = [len(lexicon[w[1]]) for w in potential]
			choice = random.choices(potential, ws)
			simplified = simplify(choice[0][1], lexicon)
			reg_index = choice[0][0]
	elif len(potential) == 1:  # One potential word to simplify
		simplified = simplify(potential[0][1], lexicon)  # returns a string (can be multiple words)
		reg_index = potential[0][0]
	else:  # No potential words found
		target_words = set(target_sent_split) if target_sent is not None else set()
		shared_words = (set(org_sent_split) & target_words) - set('.')
		simplified = random.choice(list(shared_words)) if shared_words != set() else random.choice(list(set(org_sent_split) - set('.')))
		reg_index = org_sent_split.index(simplified)
	if target_sent is not None:  # Training time:
		if simplified in target_sent:  # If simplified in sim_sentence, then find index
			sim_index = target_sent_split.index(simplified.split(' ')[0])  # index of first word of simplified
		else:  # else pick random index
			sim_index = random.randint(0, len(target_sent_split) - 1)
		return simplified, reg_index, sim_index  # simplified, index in org_sent, index in target_sent
	else:  # Test time:
		return simplified, reg_index  # simplified, index in org_sent


def get_simple_word_rand(org_sent, target_sent=None):
	if target_sent is None:  # Test time
		org_index = random.randint(0, len(org_sent.split(' '))-1)
		return org_sent.split(' ')[org_index], org_index  # simplified, index in org_sent
	else:  # Train time
		org_index = random.randint(0, len(org_sent.split(' '))-1)
		target_index = random.randint(0, len(target_sent.split(' '))-1)
		return target_sent.split(' ')[target_index], org_index, target_index  # simplified, index in org_sent, index in target_sent


def get_simple_word(org_sent, target_sent=None, get_kind=0, lex=dic_reg_to_sim):
	assert type(get_kind) == int and get_kind < 4
	if get_kind == 0:
		return get_simple_word_rand(org_sent, target_sent)
	elif get_kind == 1:
		return get_simple_word_dict(org_sent, target_sent, lexicon=lex)
	elif get_kind == 2:
		pass
		# return get_simple_word_classifier(org_sent, target_sent=None, lexicon=lex)
