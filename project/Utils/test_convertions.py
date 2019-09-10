def make_unedited(in_file, out_file):
	for line in in_file:
		if line != '\n' and line != '$$$\n':
			l = line[0]+line[1:].strip().replace("•","\n•").replace(". ", ".\n").replace('\n\n','\n')+'\n'
			out_file.write(l)
		# out_file.write(line.strip())



# in_path = "C:\\Users\\eytanc\\Documents\\test_convert_unedited.txt"
in_path = "C:\\Users\\eytanc\\Desktop\\Semi_2_Script_regular_translated.txt"
out_path = "C:\\Users\\eytanc\\Documents\\test_convert_unedited_out.txt"

with open(in_path, 'r', encoding='utf-8') as in_f, \
		 open(out_path, 'w', encoding='utf-8') as out_f:
	make_unedited(in_f, out_f)