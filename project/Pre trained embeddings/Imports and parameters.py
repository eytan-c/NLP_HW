
working_dir = ''

data_path = working_dir+'sim_dataset_23082019.csv'
lm_data_path = working_dir+'he_htb-ud-train.txt'
json_dir = working_dir+'lexicon.json'
embeddings_file = working_dir+'cc.he.300.vec'
results_dir = working_dir+'Results_Pre-trained/'


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
import io
from nltk.translate import bleu_score
from sklearn.metrics.pairwise import cosine_similarity

"""#### Parameters definitions"""

hidden_size = 300
embedding_size = 300
n_layers = 2
dropout_p = 0.05
teacher_forcing_ratio = 1
n_epochs = 30
learning_rate = 0.0003

print_every = 1
print_loss_total = 0 # Reset every print_every
train_losses = []
val_losses = []
fk_belu_train = []
fk_belu_val = []
dic_reg_to_sim = {}

train_LM = False
train_LM_n_epochs = 11
train_LM_teacher_forcing_ratio = 0.7

train_LM_data_size = 6000#00 # 6000 is everything
data_size = 6000 # 6000 is everything
entires_1_2 = True # use only entires type 1 and 2
load_from_pickle = False
pickle_python2 = False
load_model_flag = False
model_params_file = working_dir+"model_params_pre_trained_n_epochs_"+str(n_epochs)+"_LM_"+str(train_LM_n_epochs)+".tar"


time = '{date:%Y-%m-%d_%H:%M:%S}'.format( date=datetime.datetime.now() ) 
if train_LM:
  files_suffix = '_Pre-Trained_Embeddings_epoch_%d_LM_epoch_%d_time_%s' %(n_epochs,train_LM_n_epochs,time)
else:
  files_suffix = '_Pre-Trained_Embeddings_epoch_%d_time_%s' % (n_epochs,time)