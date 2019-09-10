import re

dir = "C:\\Users\\eytanc\\Desktop\\"
script_file = "Semi_2_Script_full.txt"
regular = "Semi_2_Script_regular.txt"
simple = "Semi_2_Script_simple.txt"

with open(dir+script_file, 'r', encoding='utf-8') as f, \
		 open(dir+regular, 'w', encoding='utf-8') as reg, \
		 open(dir+simple, 'w', encoding='utf-8') as sim:
	for line in f:
		if line == '\n':
			continue
		if re.match(r"\d+\.\t",line):
			reg.write(line)
			sim.write(line)
			continue
		# if re.search(r"[\u0590-\u05fe]+",line):
		# 	sim.write(line)
		# else:
		# 	reg.write(line)
		