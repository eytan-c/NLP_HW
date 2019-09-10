import pandas as pd
import json
import re
import numpy as np

data_dir = 'C:\\Users\\eytanc\\OneDrive\\Documents\\University\\2018-9\\Sem B\\NLP\\Project\\Dataset\\'
df = pd.read_csv(data_dir + 'domain_analysis.csv', encoding='utf-8')
sim_sent = df['sim_sent'].str.replace(r"([.!?])", r" \1").str.replace("[\u0590-\u05CF]+", "").str.replace(
	r"[^א-ת.!?\"]+", r" ").str.replace(r"(^|\s)\"(\w)", r"\1\2").str.replace(r"(\w)\"(\s|$)",
                                                                           r"\1\2").str.strip().str.split()
reg_sent = df['reg_sent'].str.replace(r"([.!?])", r" \1").str.replace("[\u0590-\u05CF]+", "").str.replace(
	r"[^א-ת.!?\"]+", r" ").str.replace(r"(^|\s)\"(\w)", r"\1\2").str.replace(r"(\w)\"(\s|$)",
                                                                           r"\1\2").str.strip().str.split()
dom = df['domain']
new_df = pd.DataFrame(dict(dom=dom, reg_sent=reg_sent, sim_sent=sim_sent))
domain_sets = {key: {'reg': set(), 'sim': set()} for key in new_df['dom'].unique()}
new_df['sim_sent'][313] = []
new_df['sim_sent'][314] = []
new_df['sim_sent'][448] = []
new_df['sim_sent'][449] = []
for ind, row in new_df.iterrows():
	if type(row['reg_sent']) != list:
		print('reg', ind)
	if type(row['sim_sent']) != list:
		print('sim', ind)
	domain_sets[row['dom']]['reg'] = domain_sets[row['dom']]['reg'] | set(row['reg_sent'])
	domain_sets[row['dom']]['sim'] = domain_sets[row['dom']]['sim'] | set(row['sim_sent'])

uniq_legal_reg = domain_sets['legal']['reg'] - (domain_sets['security']['reg'] |
                                                domain_sets['general info']['reg'] |
                                                domain_sets['entertainment']['reg'])
uniq_security_reg = domain_sets['security']['reg'] - (domain_sets['legal']['reg'] |
                                                      domain_sets['general info']['reg'] |
                                                      domain_sets['entertainment']['reg'])
uniq_general_reg = domain_sets['general info']['reg'] - (domain_sets['legal']['reg'] |
                                                         domain_sets['security']['reg'] |
                                                         domain_sets['entertainment']['reg'])
uniq_entertain_reg = domain_sets['entertainment']['reg'] - (domain_sets['legal']['reg'] |
                                                            domain_sets['security']['reg'] |
                                                            domain_sets['general info']['reg'])


uniq_legal_sim = domain_sets['legal']['sim'] - (domain_sets['security']['sim'] |
                                                domain_sets['general info']['sim'] |
                                                domain_sets['entertainment']['sim'])
uniq_security_sim = domain_sets['security']['sim'] - (domain_sets['legal']['sim'] |
                                                      domain_sets['general info']['sim'] |
                                                      domain_sets['entertainment']['sim'])
uniq_general_sim = domain_sets['general info']['sim'] - (domain_sets['legal']['sim'] |
                                                         domain_sets['security']['sim'] |
                                                         domain_sets['entertainment']['sim'])
uniq_entertain_sim = domain_sets['entertainment']['sim'] - (domain_sets['legal']['sim'] |
                                                            domain_sets['security']['sim'] |
                                                            domain_sets['general info']['sim'])

prints = ['legal_reg', 'security_reg', 'general_reg', 'entertain_reg',
          'legal_sim', 'security_sim', 'general_sim', 'entertain_sim']

vs = [uniq_legal_reg, uniq_security_reg, uniq_general_reg, uniq_entertain_reg,
      uniq_legal_sim, uniq_security_sim, uniq_general_sim, uniq_entertain_sim]

for i, v in enumerate(vs):
    print('%s: %s' %(prints[i], len(v)))

for key in domain_sets.keys():
	for k in domain_sets[key].keys():
		print('# in %s_%s: %s' % (key, k, len(domain_sets[key][k])))

for typ in ['reg', 'sim']:
	shared = domain_sets['legal'][typ] & domain_sets['security'][typ] & \
	         domain_sets['general info'][typ] & domain_sets['entertainment'][typ]
	print('#_shared_%s: %s' % (typ, len(shared)))
