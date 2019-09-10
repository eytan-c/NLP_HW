import re

simple_path = "C:\\Users\\eytanc\\Documents\\מסמכים מפושטים.txt"
reg_path = "C:\\Users\\eytanc\\Documents\\מסמכים לא מפושטים.txt"
# out_path = "C:\\Users\\eytanc\\Documents\\sim_dataset.csv"
out_path = "C:\\Users\\eytanc\\Documents\\sim_dataset.txt"

counts = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
entry_types = ["Basic Entries", "Expansion Entries", "Summarization Entries",
               "Complex Entries", "Deletion Entries", "Information Entries"]


def sen_length(sent):
	count = 0
	i = 0
	for s in re.split(r"\.\s+", sent.replace('\\n', '').strip()):
		count += s.split()
		i += 1
	return count / i
