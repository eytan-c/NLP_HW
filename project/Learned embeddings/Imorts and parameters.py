
working_dir = ''
data_path = working_dir+'sim_dataset_23082019.csv'
lm_data_path = working_dir+'he_htb-ud-train.txt'
eng_wiki_normal_path = working_dir+'normal.aligned'
eng_wiki_simple_path = working_dir+'simple.aligned'
results_dir = working_dir+'Results/'

import torch
import numpy
from torch import autograd, nn, optim
import torch.nn.functional as F
import re
import random
import time
import math
import pandas
from sklearn.model_selection import train_test_split
import datetime
import matplotlib # must be called here for plotting in nova
matplotlib.use('pdf') # must be called here for plotting in nova
import matplotlib.pyplot as plt
import pickle
import json
from nltk.translate import bleu_score
from nltk.corpus import cmudict

"""#### Parameters definitions"""

hidden_size = 500
embedding_size = 200
n_layers = 2
dropout_p = 0.05
teacher_forcing_ratio = 1
n_epochs = 32
learning_rate = 0.0003

print_every = 1
print_loss_total = 0 # Reset every print_every
train_losses = []
val_losses = []
fk_belu_train = []
fk_belu_val = []
dic_reg_to_sim = {}
 

train_LM = False
train_LM_n_epochs = 30
train_LM_teacher_forcing_ratio = 0.7

train_LM_data_size = 6000#00 # 6000 is everything
data_size = 6000 # 121987 is everything in eng wiki
entires_1_2 = True # use only entires type 1 and 2
load_from_pickle = False
eng_wiki = False
load_model_flag = False
model_params_file = working_dir+"model_params_n_epochs_"+str(n_epochs)+".tar"
if eng_wiki:
  json_dir = working_dir+'SPPDB_lexicon.json'
else:
  json_dir = working_dir+'lexicon.json'


time = '{date:%Y-%m-%d_%H:%M:%S}'.format( date=datetime.datetime.now() ) 
if train_LM:
  files_suffix = '_epoch_%d_LM_epoch_%d_time_%s' %(n_epochs,train_LM_n_epochs,time)
elif eng_wiki:
  files_suffix = '_English_wiki_epoch_%d_time_%s' % (n_epochs,time)
else:
  files_suffix = '_epoch_%d_time_%s' % (n_epochs,time)