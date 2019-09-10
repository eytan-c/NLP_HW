import re
import numpy as np
import pandas as pd

# simple_path = "C:\\Users\\eytanc\\Documents\\מסמכים מפושטים.txt"
# reg_path = "C:\\Users\\eytanc\\Documents\\מסמכים לא מפושטים.txt"
# out_path = "C:\\Users\\eytanc\\Documents\\sim_dataset.csv"
# out_path = "C:\\Users\\eytanc\\Documents\\sim_dataset.txt"

counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
entry_types = ["Basic Entries", "Expansion Entries", "Summarization Entries",
               "Complex Entries", "Deletion Entries", "Information Entries"]

# data_entry = {'doc_title': str, 'sub_doc_id': int, 'sentence_num': int,
#               'entry_type': int,'reg_sent': str,'sim_sent': str}

sent_list = []


def get_entry_params(simple_sent, regular_sent):
	more_than_2_reg = '. ' in regular_sent[:-2]  # logical check for several regular sentences together
	more_than_2_sim = '. ' in simple_sent[:-2]  # logical check for several simplified sentences together
	sim_null = simple_sent == '\n'  # logical check for null simple sentence
	reg_null = regular_sent == '\n'  # logical check for null regular sentence
	return sim_null, reg_null, more_than_2_sim, more_than_2_reg


def entry_type(sim_null, reg_null, more_than_2_sim, more_than_2_reg):
	if not sim_null and not reg_null:  # if both reg and sim have sentences, we are in entries of types 1-4
		if more_than_2_reg and more_than_2_sim:  # complex
			return 4
		elif more_than_2_reg and not more_than_2_sim:  # summarization
			return 3
		elif not more_than_2_reg and more_than_2_sim:  # expansion
			return 2
		else:  # basic
			return 1
	else:  # if one of the sentences is missing, we are of types 5,6
		if sim_null and not reg_null:  # deletion
			return 5
		elif not sim_null and reg_null:  # information
			return 6


with open(simple_path, 'r', encoding='utf-8') as simple, \
				open(reg_path, 'r', encoding='utf-8') as reg, \
				open(out_path, 'w', encoding='utf-8') as out:
	i = 0
	out.write('doc_title;sub_doc_id;sentence_num;entry_type;reg_sent;sim_sent\n')
	while True:
		sim_line = simple.readline()
		reg_line = reg.readline()
		if not sim_line or not reg_line:
			break
		if sim_line == '$$$\n':  ## Read all the lines of expansion sections that I didn't want to remove from the file.
			sim_line = simple.readline()
			reg_line = reg.readline()
			while sim_line != '$$$\n':
				sim_line = simple.readline()
				reg_line = reg.readline()
			continue
		
		if not '>>>>>' in sim_line and not '####' in sim_line:
			sim_null, reg_null, more_than_2_sim, more_than_2_reg = get_entry_params(sim_line, reg_line)
			case = entry_type(sim_null, reg_null, more_than_2_sim, more_than_2_reg)
			if case is not None:
				i += 1
				counts[case] += 1
				out.write('%r;%r;%r;%r;%r;%r\n' % (doc_title, sub_doc_id, i, case, reg_line, sim_line))
		elif '####' in sim_line:  # this is a new main document
			if 'END' in sim_line:
				continue
			# doc_sumup(doc_title,doc_counts,doc_lines)
			else:
				doc_title = sim_line[sim_line.find('# ') + 2:sim_line.rfind(' #')]
		elif '>>>>>' in sim_line:
			sub_doc_id = int(sim_line[sim_line.rfind('>') + 1:].strip())
		# sub_doc_sumup()
	
	print("Total Entries:", i, "; Valid Entries:", counts[1] + counts[2], '\n')
	for k in sorted(counts.keys()):
		print(entry_types[k - 1] + ":", counts[k])

"""
Edit lines in bulk:
with open(test_path,'r', encoding='utf-8') as f, open(test_path[:-13]+"test_read_out.txt","w",encoding='utf-8') as o:
	for line in f:
		if "." not in line and len(line.split(" ")) < 3:
			o.write(line.replace("(", ")").replace(" )", " (").replace('\t',' '))
		else:
			o.write(line.replace("(", ")").replace(" )", " (").replace('\t',' ').replace("\n", " ").replace(". ", ".\n"))
"""
