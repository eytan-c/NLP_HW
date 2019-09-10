
"""####Complex words simplifier"""

# # TODO
# import random
# def get_simple_word(org_sent, target_sent=None):
#   if target_sent is None: # Test time
#     org_index = random.randint(0,len(org_sent.split(' '))-1)
#     return org_sent.split(' ')[org_index],org_index # simplified, index in org_sent
#   else: # Train time
#     org_index = random.randint(0,len(org_sent.split(' '))-1)
#     target_index = random.randint(0,len(target_sent.split(' '))-1)
#     return target_sent.split(' ')[target_index],org_index,target_index # simplified, index in org_sent, index in target_sent

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
		else:
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
