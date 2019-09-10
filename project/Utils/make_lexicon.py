from NLP_HW.project.get_simple_words import *


def get_entry_diffs(entries):
	entry_word_set = {}
	for i, en in enumerate(entries):
		if en[0] != ' ' and en[1] != ' ':
			reg_words = set(en[0].split(' '))
			sim_words = set(en[1].split(' '))
			reg_diff = reg_words - sim_words
			sim_diff = sim_words - reg_words
			remain_reg = [(j, word) for j, word in enumerate(en[0].split(' ')) if word in reg_diff]
			remain_sim = [(j, word) for j, word in enumerate(en[1].split(' ')) if word in sim_diff]
			entry_word_set[i] = {'reg_words': remain_reg, 'sim_words': remain_sim}
	return entry_word_set


def build_vocabs(pairs, input_vocab, output_vocab):
	for pair in pairs:
		if pair[0] != ' ' and pair[1] != ' ':  # delete empty pairs
			input_vocab.index_sentence(pair[0])
			output_vocab.index_sentence(pair[1])


def get_vocab_diffs(input_vocab, output_vocab):
	shared = input_vocab.word2index.keys() & output_vocab.word2index.keys()
	in_not_out = input_vocab.word2index.keys() - output_vocab.word2index.keys()
	out_not_in = output_vocab.word2index.keys() - input_vocab.word2index.keys()
	return shared, in_not_out, out_not_in


def get_entry_set(word_dict):
	words = set()
	for i in word_dict.keys():
		words = words | {word[1] for word in word_dict[i]['reg_words']}
	return words


entries = prepare_data('normal_heb', 'simple_heb')

reg_heb = Vocab("Heb-reg")
sim_heb = Vocab("Heb-simple")

build_vocabs(entries, reg_heb, sim_heb)

shared_words_vocab, reg_not_sim_vocab, sim_not_reg_vocab = get_vocab_diffs(reg_heb, sim_heb)
entry_words_dict = get_entry_diffs(entries)
entry_words = get_entry_set(entry_words_dict)
entry_AND_reg = entry_words & reg_not_sim_vocab
entry_NOT_reg = entry_words - reg_not_sim_vocab
reg_NOT_entry = reg_not_sim_vocab - entry_words


def build_match(entries, entry_word_dict, prog_file=data_dir+'lexicon_prog.txt', save_json=data_dir+'lexicon.json'):
	import json
	lexicon = {}
	try:
		with open(save_json, 'r', encoding='utf-8') as f:
			lexicon = json.load(f)
	except:
		print("No JSON file, will create after 1st entry")
	i = 0
	try:
		with open(prog_file, 'r', encoding='utf-8') as f:
			i = int(f.readlines()[0])
	except:
		print('No progress file, starting from i=%s' % i)
	while i < len(entries):
		if i in entry_word_dict.keys():
			print('Regular: \n %s\n' % entries[i][0])
			print('Simplified: \n %s\n' % entries[i][1])
			print('Words not in Sim:\n%s' % entry_word_dict[i]['reg_words'])
			print('Words not in Reg:\n%s' % entry_word_dict[i]['sim_words'])
			simplify = True if input('Are there any words to simplify? (y,n)') == 'y' else False
			if simplify:
				# print('Which words to simplify?')
				words_indices = []
				while True:
					reg_index = int(input('What is the index of the word to Simplify?'))
					sim_index = tuple(map(int, input('What is the index of the simplification? start,end').split(',')))
					words_indices.append((reg_index, sim_index))
					more_words = True if input('Are there any more words (y,n)?') == 'y' else False
					if not more_words:
						break
				for index in words_indices:
					print(index)
					key = entries[i][0].split(' ')[index[0]]
					# There is a possibility of several different simplifications
					if key not in lexicon.keys():
						lexicon[key] = []
					if len(index[1]) == 1:
						lexicon[key].append((i, entries[i][1].split(' ')[index[1][0]])) # saving the entry which came from
						# lexicon[key].append(entries[i][1].split(' ')[index[1][0]])  # not saving entry which came from
					else:
						lexicon[key].append((i, tuple(entries[i][1].split(' ')[index[1][0]:index[1][1]+1])))  # saving the entry which came from
						# lexicon[key].append(tuple(entries[i][1].split(' ')[index[1][0]:index[1][1] + 1]))  # not saving entry which came from
				with open(prog_file, 'w', encoding='utf-8') as f:
					print('Saving progress...')
					f.write(str(i+1))
				with open(save_json, 'w', encoding='utf-8') as d:
					print('Saving JSON...')
					json.dump(lexicon, d, ensure_ascii=False, sort_keys=True)
		i += 1
	return lexicon
