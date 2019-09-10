#### Vocabulary
"""

SOS_TOKEN = 0
EOS_TOKEN = 1
UNKNOWN = 2
MAX_LENGTH = 20

#class Vocab:
class Vocab(object):
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

"""#### Embeddings"""

def load_embeddings(embeddings_file):
    fin = io.open(embeddings_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(numpy.int32, fin.readline().split())
    embeddings = {}
    #all_vocabs_words = [unicode(word, "utf-8") for word in input_vocab.word2index.keys()+output_vocab.word2index.keys()]
    for line in fin:
        tokens = line.rstrip().split(' ') # token[0] is the word itself        
        #if tokens[0] in all_vocabs_words:
        if (tokens[0] in input_vocab.word2index) or (tokens[0] in output_vocab.word2index):
          embeddings[tokens[0]] = torch.tensor(list(map(numpy.float32, tokens[1:]))).unsqueeze(0)
#         if tokens[0] in '.':
#           embeddings['<EOS>'] = torch.tensor(list(map(numpy.float32, tokens[1:]))).unsqueeze(0)
        if tokens[0] in '</s>':
          embeddings['<SOS>'] = torch.tensor(list(map(numpy.float32, tokens[1:]))).unsqueeze(0)
          embeddings['<EOS>'] = -1*torch.tensor(list(map(numpy.float32, tokens[1:]))).unsqueeze(0)
        del tokens
    embeddings['unknown'] = torch.ones([300]).unsqueeze(0)/1000
    return embeddings

def get_embeddings(word):
  if word in embeddings:
    return embeddings[word]
  else:
    #print('#'+word+'#') # All words suppose to have embeddings
    return embeddings['unknown'] # TODO : to check if there is an uknown word, and if not enter one
  
def get_sentence_embeddings(sent):
  emd_list = []
  for word in sent.rstrip().split(' '):
    emd_list.append(get_embeddings(word))
  result = torch.Tensor(len(emd_list), emd_list[0].shape[0])
  return torch.cat(emd_list, out=result)

def get_word_from_embeddings(word_embedding):
#   max_similarity = -float('inf')
#   closest_word = 'NO CLOSE WORD'
#   for cur_word, cur_emd in embeddings.items():
#     cur_simi = F.cosine_similarity(word_embedding,cur_emd.cuda())
#     if cur_simi > max_similarity:
#       max_similarity = cur_simi
#       closest_word = cur_word
#   return closest_word
  word_embedding = numpy.asarray(word_embedding.detach().cpu())
  all_emd = numpy.asarray([numpy.asarray(emd.squeeze(0)) for emd in embeddings.values()])
  cos_sim_mat = cosine_similarity(word_embedding,all_emd)
  max_sim_list = numpy.argmax(cos_sim_mat, axis=1)
  all_words = list(embeddings.keys())
  return all_words[max_sim_list[0]]


def get_embeddings_similarity_dict(embeddings):
  dic = {}
  from sklearn.metrics.pairwise import cosine_similarity
  all_emd = numpy.asarray([numpy.asarray(emd.squeeze(0)) for emd in embeddings.values()])
  cos_sim_mat = cosine_similarity(all_emd,all_emd)
  #max_sim_list = numpy.argmax(cos_sim_mat, axis=1)
  max_sim_list = numpy.argsort(cos_sim_mat, axis=1)[:,-2] # get second argmax because first argmax is the word itself
  
  all_words = list(embeddings.keys())
  for i, word in enumerate(all_words):
    dic[word] = all_words[max_sim_list[i]]
  return dic

def reverse_and_slice_tensor(tensor, slicing_index):
  idx = [i for i in range(tensor.size(0)-1, -1, -1)]
  idx = torch.LongTensor(idx[:len(idx)-1-slicing_index])
  return tensor.index_select(0, idx)

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
        
    if 'simple-wiki'==lang1 and 'normal-wiki'==lang2:
      # Read the file and split into lines
      lines = open('/content/drive/My Drive/Colab Notebooks/nlp/data/%s-%s.txt' % (lang1, lang2)).read().strip().split('\n')

      # Split every line into pairs and normalize
      lines = [line.split('\t')[2] for line in lines]
      pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

      pairs = [pairs[i]+pairs[i+1] for i in range(0,len(lines),2)]
      
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
  with open(json_dir,'r') as f:
	   dic_reg_to_sim = json.load(f)
  
  print("Reading lines...")
  if train_LM:
    #pairs = prepare_data('lm_train_heb', 'lm_train_heb')
    pairs = read_langs('lm_train_heb', 'lm_train_heb')
    print("Read %s sentence pairs for LM" % len(pairs))    
    pairs = filter_pairs(pairs)
    X_lm, y_lm = pairs_to_data(pairs, train_LM_data_size)

  pairs = read_langs('normal_heb', 'simple_heb')
  print("Read %s sentence pairs" % len(pairs))    
  pairs = filter_pairs(pairs)
  X, y = pairs_to_data(pairs, data_size)
  #dic_reg_to_sim = build_dic_reg_to_sim()
  print("Sentences pairs left:%s" % len(X))
  
  print("Loading word embeddings...")
  emd = load_embeddings(embeddings_file)
  
  print("Splitting data to train, validtion and test")
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=1)
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1)

  print("Train size:",len(X_train))
  print("Validation size:",len(X_val))
  print("Test size:",len(X_test))

  print("Data is ready!")
  
  if train_LM:
    return emd, input_vocab, output_vocab, dic_reg_to_sim, X_train, y_train, X_val, y_val, X_test, y_test, X_lm, y_lm
  else:
    return emd, input_vocab, output_vocab, dic_reg_to_sim, X_train, y_train, X_val, y_val, X_test, y_test
