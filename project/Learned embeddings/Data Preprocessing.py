
#### Vocabulary
"""

SOS_TOKEN = 0
EOS_TOKEN = 1
UNKNOWN = 2
MAX_LENGTH = 30

class Vocab:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.index2word = {UNKNOWN:'__unk__'}
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

def normalize_string(s,only_heb=False):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    if only_heb:
      s = re.sub(r"[\u0590-\u05CF]+", "", s)
      s = re.sub(r"[^?-?.!?]+", r" ", s)
      s = re.sub(r"(^|\s)\"(\w)", r"\1\2", re.sub(r"(\w)\"(\s|$)", r"\1\2", s))
    else:
      s = re.sub(r"[^a-zA-Z?-?.!?]+", r" ", s)
    return s.strip()
  
def filter_pair(p):
    not_too_long = len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH
    not_too_short = len(p[0].split(' ')) > 1 and len(p[1].split(' ')) > 1
    return not_too_long and not_too_short

def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]

def read_langs(lang1, lang2):    
    if 'normal_heb' == lang1 or 'simple_heb' == lang2:
      df = pandas.read_csv(data_path, error_bad_lines=False)#, encoding='utf-8')
      if entires_1_2:
        df = df.loc[df['entry_type']<3]
      all_reg_sent = df['reg_sent']
      all_sim_sent = df['sim_sent']

      pairs = []
      for i in list(df.index.values):
        pairs.append([normalize_string(all_reg_sent[i],only_heb=True),normalize_string(all_sim_sent[i],only_heb=True)])
        
    if 'small_simple-wiki'==lang1 and 'small_normal-wiki'==lang2:
      # Read the file and split into lines
      lines = open('/content/drive/My Drive/Colab Notebooks/nlp/data/%s-%s.txt' % (lang1, lang2)).read().strip().split('\n')

      # Split every line into pairs and normalize
      lines = [line.split('\t')[2] for line in lines]
      pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

      pairs = [pairs[i]+pairs[i+1] for i in range(0,len(lines),2)]
      
    if 'normal-wiki'==lang1 and 'simple-wiki'==lang2:
      # Read the files and split into lines
      normal_lines = open(eng_wiki_normal_path).read().strip().split('\n')
      simple_lines = open(eng_wiki_simple_path).read().strip().split('\n')
      
      # Split every line into pairs and normalize
      normal_lines = [line.split('\t')[2] for line in normal_lines]
      simple_lines = [line.split('\t')[2] for line in simple_lines]
      
      normal_lines = [[normalize_string(s) for s in l.split('\t')] for l in normal_lines]
      simple_lines = [[normalize_string(s) for s in l.split('\t')] for l in simple_lines]

      pairs = [normal_lines[i]+simple_lines[i] for i in range(len(normal_lines))]   
      
    if 'lm_train_heb'==lang1 and 'lm_train_heb'==lang2:
      lines = open(lm_data_path).read().strip().replace('\n', ' ').split('. ')
      pairs = [[normalize_string(line,only_heb=True),normalize_string(line,only_heb=True)] for line in lines]
        
    return pairs


def print_pair(p):
    print(p[0])
    print(p[1])

def pairs_to_data(pairs, size):
  X = []
  y = []

  for pair in pairs[:size]:
    if pair[0]!=' ' and pair[1]!= ' ': # delete empty pairs
      input_vocab.index_sentence(pair[0])
      output_vocab.index_sentence(pair[1])
      X.append(pair[0])
      y.append(pair[1])
  
  print("Trimmed to %s non-empty sentence pairs" % len(X))

  # Print example
  i = random.randint(0,len(X)-1)
  print(X[i])
  print(y[i])
  return numpy.asarray(X), numpy.asarray(y)

"""#### Init data"""

def init_data(input_vocab,output_vocab,dic_reg_to_sim):
  #input_vocab, output_vocab, pairs = prepare_data('simple-wiki', 'normal-wiki', True)
  
  print("Readin lexicon...")
  with open(json_dir,'r') as f: #,encoding='utf-8'
	   dic_reg_to_sim = json.load(f)
  
  print("Reading lines...")
  if train_LM:
    #pairs = prepare_data('lm_train_heb', 'lm_train_heb')
    pairs = read_langs('lm_train_heb', 'lm_train_heb')
    print("Read %s sentence pairs for LM" % len(pairs))    
    pairs = filter_pairs(pairs)
    X_lm, y_lm = pairs_to_data(pairs, train_LM_data_size)

  if eng_wiki:
    print("Data is in English from wikipedia")
    pairs = read_langs('normal-wiki', 'simple-wiki')
  else:
    print("Data is in Hebrew from our dataset")
    pairs = read_langs('normal_heb', 'simple_heb')
  print("Read %s sentence pairs" % len(pairs))    
  pairs = filter_pairs(pairs)
  X, y = pairs_to_data(pairs, data_size)
  #dic_reg_to_sim = build_dic_reg_to_sim()
  print("Sentences pairs left:%s" % len(X))

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=1)
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1)

  del X,y,pairs
  
  print("Train size:",len(X_train))
  print("Validation size:",len(X_val))
  print("Test size:",len(X_test))

  print("Data is ready!")
  if train_LM:
    return input_vocab, output_vocab, dic_reg_to_sim, X_train, y_train, X_val, y_val, X_test, y_test, X_lm, y_lm
  else:
    return input_vocab, output_vocab, dic_reg_to_sim, X_train, y_train, X_val, y_val, X_test, y_test