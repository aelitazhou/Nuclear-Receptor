## importing part
from __future__ import division, print_function, absolute_import
import statsmodels.api as sm
import gzip
import os
import re
import tarfile
import math
import random
import sys
import time
import logging
import numpy as np
import math

from scipy.sparse import coo_matrix
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn import preprocessing
from tensorflow.python.platform import gfile
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d,conv_1d,max_pool_1d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge
#from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell,GRUCell
#from recurrent import bidirectional_rnn, BasicLSTMCell,GRUCell

random.seed(1234)
#### data and vocabulary

data_dir="./data"
vocab_size_compound=68
vocab_size_protein=76
comp_MAX_size=100
protein_MAX_size=152
vocab_compound="vocab_compound"
vocab_protein="vocab_protein"
batch_size = 64

GRU_size_prot=256
GRU_size_drug=128

dev_perc=0.2

#arguments
learning_rate = float(sys.argv[1]) 
drop_out = float(sys.argv[2])   
num1_neurons = int(sys.argv[3]) 
num2_neurons = int(sys.argv[4]) 

## Padding part
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
_WORD_SPLIT = re.compile(b"(\S)")
_WORD_SPLIT_2 = re.compile(b",")
_DIGIT_RE = re.compile(br"\d")


## functions
def basic_tokenizer(sentence,condition):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    if condition ==0:
        l = _WORD_SPLIT.split(space_separated_fragment)
        del l[0::2]
    elif condition == 1:
        l = _WORD_SPLIT_2.split(space_separated_fragment)
    words.extend(l)
  return [w for w in words if w]

def sentence_to_token_ids(sentence, vocabulary,condition,normalize_digits=False):

  words = basic_tokenizer(sentence,condition)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]


def initialize_vocabulary(vocabulary_path):
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)



def data_to_token_ids(data_path, target_path, vocabulary_path, condition,normalize_digits=False):
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(tf.compat.as_bytes(line), vocab, condition,normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def read_data(source_path,MAX_size):
  data_set = []
  mycount=0
  with tf.gfile.GFile(source_path, mode="r") as source_file:
      source = source_file.readline()
      counter = 0
      while source:
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        if len(source_ids) < MAX_size:
           pad = [PAD_ID] * (MAX_size - len(source_ids))
           data_set.append(list(source_ids + pad))
           mycount=mycount+1
        elif len(source_ids) == MAX_size:
           data_set.append(list(source_ids))
           mycount=mycount+1
        else:
           print("there is a data with length bigger than the max\n")
           print(len(source_ids))
        source = source_file.readline()
  return data_set

def prepare_data(data_dir, train_path, vocabulary_size,vocab,max_size,condition):
  vocab_path = os.path.join(data_dir, vocab)

  train_ids_path = train_path + (".ids%d" % vocabulary_size)
  data_to_token_ids(train_path, train_ids_path, vocab_path,condition)
  train_set = read_data(train_ids_path,max_size)
  
  return train_set

def read_labels(path):
    x = []
    f = open(path, "r") 
    for line in f:
         if (line[0]=="<")or(line[0]==">"): 
            print("Inequality in IC50!!!\n")
         else:
            line = [float(i) for i in line.strip('\n').split(' ')]
            x.append(line)
            #x.append(float(line)) 
 
    #y = normalize_labels(x)
    return np.array(x)


def read_initial_state_weigths(path,size1,size2):
    x = []
    f = open(path, "r")
    count = 0;
    for line in f:
       y = [float(n) for n in line.split(" ")]
       if len(y) == size2:
          x.append(y)
          count = count+1
       else:
          print("not exactly equal to size2!!!!!!")
    
    return x

def  train_dev_split(train_protein,train_compound,train_IC50,dev_perc,comp_MAX_size,protein_MAX_size,batch_size):
    num_whole= len(train_IC50)
    num_train = math.ceil(num_whole*(1-dev_perc)/batch_size)*batch_size
    num_dev = math.floor((num_whole - num_train)/batch_size)*batch_size

    index_total = range(0,num_whole)
    index_dev = sorted(random.sample(index_total,num_dev))
    remain = list(set(index_total)^set(index_dev))
    index_train = sorted(random.sample(remain,num_train))

    compound_train = [train_compound[i] for i in index_train]
    compound_train = np.reshape(compound_train,[len(compound_train),comp_MAX_size])
    compound_dev = [train_compound[i] for i in index_dev]
    compound_dev = np.reshape(compound_dev,[len(compound_dev),comp_MAX_size])

    IC50_train = [train_IC50[i] for i in index_train]
    IC50_train = np.reshape(IC50_train,[len(IC50_train),5])
    IC50_dev = [train_IC50[i] for i in index_dev]
    IC50_dev = np.reshape(IC50_dev,[len(IC50_dev),5])

    protein_train = [train_protein[i] for i in index_train]
    protein_train = np.reshape(protein_train,[len(protein_train),protein_MAX_size])
    protein_dev = [train_protein[i] for i in index_dev]
    protein_dev = np.reshape(protein_dev,[len(protein_dev),protein_MAX_size])

    return compound_train, compound_dev, IC50_train, IC50_dev, protein_train, protein_dev


def diagnosis(l_true, l_pred):
    true = []
    pred = []
    for i in range(l_true.shape[0]):
        for j in range(l_true.shape[1]):
            if l_true[i, j] == 1:
                true.append(j)
            if l_pred[i, j] == 1:
                pred.append(j)
    a = confusion_matrix(true, pred)
    print(a)
    acc = []
    for i in range(a.shape[0]):
        acc.append(a[i, i]/np.sum(a[i, :]))
    print(acc)
    print(np.mean(acc))


################ Reading initial states and weigths 
prot_gru_1_candidate_bias_init = read_initial_state_weigths("./data/prot_init/cell_0_candidate_bias.txt",1,GRU_size_prot)
prot_gru_1_candidate_bias_init = tf.convert_to_tensor(np.reshape(prot_gru_1_candidate_bias_init,[GRU_size_prot]),dtype=tf.float32)

prot_gru_1_candidate_kernel_init = read_initial_state_weigths("./data/prot_init/cell_0_candidate_kernel.txt",2*GRU_size_prot,GRU_size_prot)
prot_gru_1_candidate_kernel_init = tf.convert_to_tensor(np.reshape(prot_gru_1_candidate_kernel_init,[2*GRU_size_prot,GRU_size_prot]),dtype=tf.float32)

prot_gru_1_gates_bias_init = read_initial_state_weigths("./data/prot_init/cell_0_gates_bias.txt",1,2*GRU_size_prot)
prot_gru_1_gates_bias_init = tf.convert_to_tensor(np.reshape(prot_gru_1_gates_bias_init,[2*GRU_size_prot]),dtype=tf.float32)

prot_gru_1_gates_kernel_init = read_initial_state_weigths("./data/prot_init/cell_0_gates_kernel.txt",2*GRU_size_prot,2*GRU_size_prot)
prot_gru_1_gates_kernel_init = tf.convert_to_tensor(np.reshape(prot_gru_1_gates_kernel_init,[2*GRU_size_prot,2*GRU_size_prot]),dtype=tf.float32)

prot_gru_2_candidate_bias_init = read_initial_state_weigths("./data/prot_init/cell_1_candidate_bias.txt",1,GRU_size_prot)
prot_gru_2_candidate_bias_init = tf.convert_to_tensor(np.reshape(prot_gru_2_candidate_bias_init,[GRU_size_prot]),dtype=tf.float32)

prot_gru_2_candidate_kernel_init = read_initial_state_weigths("./data/prot_init/cell_1_candidate_kernel.txt",2*GRU_size_prot,GRU_size_prot)
prot_gru_2_candidate_kernel_init = tf.convert_to_tensor(np.reshape(prot_gru_2_candidate_kernel_init,[2*GRU_size_prot,GRU_size_prot]),dtype=tf.float32)

prot_gru_2_gates_bias_init = read_initial_state_weigths("./data/prot_init/cell_1_gates_bias.txt",1,2*GRU_size_prot)
prot_gru_2_gates_bias_init = tf.convert_to_tensor(np.reshape(prot_gru_2_gates_bias_init,[2*GRU_size_prot]),dtype=tf.float32)

prot_gru_2_gates_kernel_init = read_initial_state_weigths("./data/prot_init/cell_1_gates_kernel.txt",2*GRU_size_prot,2*GRU_size_prot)
prot_gru_2_gates_kernel_init = tf.convert_to_tensor(np.reshape(prot_gru_2_gates_kernel_init,[2*GRU_size_prot,2*GRU_size_prot]),dtype=tf.float32)

prot_embd_init = read_initial_state_weigths("./data/prot_init/embedding_W.txt",vocab_size_protein,GRU_size_prot)
prot_embd_init = tf.convert_to_tensor(np.reshape(prot_embd_init,[vocab_size_protein,GRU_size_prot]),dtype=tf.float32)

prot_init_state_1 = read_initial_state_weigths("./data/prot_init/first_layer_states.txt",batch_size,GRU_size_prot)
prot_init_state_1 = tf.convert_to_tensor(np.reshape(prot_init_state_1,[batch_size,GRU_size_prot]),dtype=tf.float32)

prot_init_state_2 = read_initial_state_weigths("./data/prot_init/second_layer_states.txt",batch_size,GRU_size_prot)
prot_init_state_2 = tf.convert_to_tensor(np.reshape(prot_init_state_2,[batch_size,GRU_size_prot]),dtype=tf.float32)


drug_gru_1_candidate_bias_init = read_initial_state_weigths("./data/drug_init/cell_0_candidate_bias.txt",1,GRU_size_drug)
drug_gru_1_candidate_bias_init = tf.convert_to_tensor(np.reshape(drug_gru_1_candidate_bias_init,[GRU_size_drug]),dtype=tf.float32)

drug_gru_1_candidate_kernel_init = read_initial_state_weigths("./data/drug_init/cell_0_candidate_kernel.txt",2*GRU_size_drug,GRU_size_drug)
drug_gru_1_candidate_kernel_init = tf.convert_to_tensor(np.reshape(drug_gru_1_candidate_kernel_init,[2*GRU_size_drug,GRU_size_drug]),dtype=tf.float32)

drug_gru_1_gates_bias_init = read_initial_state_weigths("./data/drug_init/cell_0_gates_bias.txt",1,2*GRU_size_drug)
drug_gru_1_gates_bias_init = tf.convert_to_tensor(np.reshape(drug_gru_1_gates_bias_init,[2*GRU_size_drug]),dtype=tf.float32)

drug_gru_1_gates_kernel_init = read_initial_state_weigths("./data/drug_init/cell_0_gates_kernel.txt",2*GRU_size_drug,2*GRU_size_drug)
drug_gru_1_gates_kernel_init = tf.convert_to_tensor(np.reshape(drug_gru_1_gates_kernel_init,[2*GRU_size_drug,2*GRU_size_drug]),dtype=tf.float32)

drug_gru_2_candidate_bias_init = read_initial_state_weigths("./data/drug_init/cell_1_candidate_bias.txt",1,GRU_size_drug)
drug_gru_2_candidate_bias_init = tf.convert_to_tensor(np.reshape(drug_gru_2_candidate_bias_init,[GRU_size_drug]),dtype=tf.float32)

drug_gru_2_candidate_kernel_init = read_initial_state_weigths("./data/drug_init/cell_1_candidate_kernel.txt",2*GRU_size_drug,GRU_size_drug)
drug_gru_2_candidate_kernel_init = tf.convert_to_tensor(np.reshape(drug_gru_2_candidate_kernel_init,[2*GRU_size_drug,GRU_size_drug]),dtype=tf.float32)

drug_gru_2_gates_bias_init = read_initial_state_weigths("./data/drug_init/cell_1_gates_bias.txt",1,2*GRU_size_drug)
drug_gru_2_gates_bias_init = tf.convert_to_tensor(np.reshape(drug_gru_2_gates_bias_init,[2*GRU_size_drug]),dtype=tf.float32)

drug_gru_2_gates_kernel_init = read_initial_state_weigths("./data/drug_init/cell_1_gates_kernel.txt",2*GRU_size_drug,2*GRU_size_drug)
drug_gru_2_gates_kernel_init = tf.convert_to_tensor(np.reshape(drug_gru_2_gates_kernel_init,[2*GRU_size_drug,2*GRU_size_drug]),dtype=tf.float32)

drug_embd_init = read_initial_state_weigths("./data/drug_init/embedding_W.txt",vocab_size_compound,GRU_size_drug)
drug_embd_init = tf.convert_to_tensor(np.reshape(drug_embd_init,[vocab_size_compound,GRU_size_drug]),dtype=tf.float32)

drug_init_state_1 = read_initial_state_weigths("./data/drug_init/first_layer_states.txt",batch_size,GRU_size_drug)
drug_init_state_1 = tf.convert_to_tensor(np.reshape(drug_init_state_1,[batch_size,GRU_size_drug]),dtype=tf.float32)

drug_init_state_2 = read_initial_state_weigths("./data/drug_init/second_layer_states.txt",batch_size,GRU_size_drug)
drug_init_state_2 = tf.convert_to_tensor(np.reshape(drug_init_state_2,[batch_size,GRU_size_drug]),dtype=tf.float32)

train_protein = prepare_data(data_dir,"./data/train_final_sps",vocab_size_protein,vocab_protein,protein_MAX_size,1)
train_compound = prepare_data(data_dir,"./data/train_final_smile",vocab_size_compound,vocab_compound,comp_MAX_size,0)
train_IC50 = read_labels("./data/train_final_ic50")

test0_protein = prepare_data(data_dir,"./data/test0_sps",vocab_size_protein,vocab_protein,protein_MAX_size,1)
test0_compound = prepare_data(data_dir,"./data/test0_smile",vocab_size_compound,vocab_compound,comp_MAX_size,0)
test0_IC50 = read_labels("./data/test0_ic50")

test1_protein = prepare_data(data_dir,"./data/test1_sps",vocab_size_protein,vocab_protein,protein_MAX_size,1)
test1_compound = prepare_data(data_dir,"./data/test1_smile",vocab_size_compound,vocab_compound,comp_MAX_size,0)
test1_IC50 = read_labels("./data/test1_ic50")

test2_protein = prepare_data(data_dir,"./data/test2_sps",vocab_size_protein,vocab_protein,protein_MAX_size,1)
test2_compound = prepare_data(data_dir,"./data/test2_smile",vocab_size_compound,vocab_compound,comp_MAX_size,0)
test2_IC50 = read_labels("./data/test2_ic50")

test3_protein = prepare_data(data_dir,"./data/test3_sps",vocab_size_protein,vocab_protein,protein_MAX_size,1)
test3_compound = prepare_data(data_dir,"./data/test3_smile",vocab_size_compound,vocab_compound,comp_MAX_size,0)
test3_IC50 = read_labels("./data/test3_ic50")


## separating train,dev, test data
compound_train, compound_dev, IC50_train, IC50_dev, protein_train, protein_dev = train_dev_split(train_protein,train_compound,train_IC50,dev_perc,comp_MAX_size,protein_MAX_size,batch_size)

## RNN for protein
prot_data = input_data(shape=[None, protein_MAX_size])
prot_embd = tflearn.embedding(prot_data, input_dim=vocab_size_protein, output_dim=GRU_size_prot)
prot_gru_1 = tflearn.gru(prot_embd, GRU_size_prot,initial_state= prot_init_state_1,trainable=True,return_seq=True,restore=False)
prot_gru_1 = tf.stack(prot_gru_1,axis=1)
prot_gru_2 = tflearn.gru(prot_gru_1, GRU_size_prot,initial_state= prot_init_state_2,trainable=True,return_seq=True,restore=False)
prot_gru_2 = tf.stack(prot_gru_2,axis=1)

drug_data = input_data(shape=[None, comp_MAX_size])
drug_embd = tflearn.embedding(drug_data, input_dim=vocab_size_compound, output_dim=GRU_size_drug)
drug_gru_1 = tflearn.gru(drug_embd,GRU_size_drug,initial_state= drug_init_state_1,trainable=True,return_seq=True,restore=False)
drug_gru_1 = tf.stack(drug_gru_1,1)
drug_gru_2 = tflearn.gru(drug_gru_1, GRU_size_drug,initial_state= drug_init_state_2,trainable=True,return_seq=True,restore=False)
drug_gru_2 = tf.stack(drug_gru_2,axis=1)


W = tflearn.variables.variable(name="Attn_W_prot",shape=[GRU_size_prot,GRU_size_drug],initializer=tf.random_normal([GRU_size_prot,GRU_size_drug],stddev=0.1),restore=False)
b = tflearn.variables.variable(name="Attn_b_prot",shape=[protein_MAX_size,comp_MAX_size],initializer=tf.random_normal([protein_MAX_size,comp_MAX_size],stddev=0.1),restore=False)
alphas_pair = tf.einsum('ij,bki->bkj',W,prot_gru_2)
alphas_pair = tf.tanh(tf.einsum('bkj,bsj->bks',alphas_pair,drug_gru_2) + b)
alphas_pair = tflearn.reshape(alphas_pair,[-1,comp_MAX_size*protein_MAX_size])
alphas_pair = tf.nn.softmax(alphas_pair,name='alphas')
alphas_pair = tflearn.reshape(alphas_pair,[-1,protein_MAX_size,comp_MAX_size])

U_size = 256
U_prot = tflearn.variables.variable(name="Attn_U_prot",shape=[U_size,GRU_size_prot],initializer=tf.random_normal([U_size,GRU_size_prot],stddev=0.1),restore=False)
U_drug = tflearn.variables.variable(name="Attn_U_drug",shape=[U_size,GRU_size_drug],initializer=tf.random_normal([U_size,GRU_size_drug],stddev=0.1),restore=False)
B = tflearn.variables.variable(name="Attn_B",shape=[U_size],initializer=tf.random_normal([U_size],stddev=0.1),restore=False)

space_1 = tf.einsum('ij,bsj->bsi',U_prot,prot_gru_2)
space_2 = tf.einsum('ij,bsj->bsi',U_drug,drug_gru_2)
Attn_space =  tf.tanh(tf.einsum('bik,bjk->bijk',space_1,space_2) + B)
Attn = tf.einsum('bijk,bij->bk',Attn_space,alphas_pair)

Attn_reshape = tflearn.reshape(Attn, [-1, U_size,1])
conv_1 = conv_1d(Attn_reshape, 64, 4,2, activation='leakyrelu', weights_init="xavier",regularizer="L2",name='conv1')
pool_1 = max_pool_1d(conv_1, 4,name='pool1')
#conv_2 = conv_1d(pool_1, 64, 4,2, activation='leakyrelu', weights_init="xavier",regularizer="L2",name='conv2')
#pool_2 = max_pool_1d(conv_2, 4,name='pool2')

pool_2 = tflearn.reshape(pool_1, [-1, 64*32])



prot_embd_W = []
prot_gru_1_gate_matrix = []
prot_gru_1_gate_bias = []
prot_gru_1_candidate_matrix = []
prot_gru_1_candidate_bias = []
prot_gru_2_gate_matrix = []
prot_gru_2_gate_bias = []
prot_gru_2_candidate_matrix = []
prot_gru_2_candidate_bias = []
for v in tf.global_variables():
   if "GRU/GRU/GRUCell/Gates/Linear/Matrix" in v.name :
      prot_gru_1_gate_matrix.append(v)
   elif "GRU/GRU/GRUCell/Candidate/Linear/Matrix" in v.name :
      prot_gru_1_candidate_matrix.append(v)
   elif "GRU/GRU/GRUCell/Gates/Linear/Bias" in v.name :
      prot_gru_1_gate_bias.append(v)
   elif "GRU/GRU/GRUCell/Candidate/Linear/Bias" in v.name :
      prot_gru_1_candidate_bias.append(v)
   elif "GRU_1/GRU_1/GRUCell/Gates/Linear/Matrix" in v.name :
      prot_gru_2_gate_matrix.append(v)
   elif "GRU_1/GRU_1/GRUCell/Candidate/Linear/Matrix" in v.name :
      prot_gru_2_candidate_matrix.append(v)
   elif "GRU_1/GRU_1/GRUCell/Gates/Linear/Bias" in v.name :
      prot_gru_2_gate_bias.append(v)
   elif "GRU_1/GRU_1/GRUCell/Candidate/Linear/Bias" in v.name :
      prot_gru_2_candidate_bias.append(v)
   elif "Embedding" in v.name:
      prot_embd_W.append(v)


drug_embd_W = []
drug_gru_1_gate_matrix = []
drug_gru_1_gate_bias = []
drug_gru_1_candidate_matrix = []
drug_gru_1_candidate_bias = []
drug_gru_2_gate_matrix = []
drug_gru_2_gate_bias = []
drug_gru_2_candidate_matrix = []
drug_gru_2_candidate_bias = []
for v in tf.global_variables():
   print(v)
   if "GRU_2/GRU_2/GRUCell/Gates/Linear/Matrix" in v.name :
      drug_gru_1_gate_matrix.append(v)
   elif "GRU_2/GRU_2/GRUCell/Candidate/Linear/Matrix" in v.name :
      drug_gru_1_candidate_matrix.append(v)
   elif "GRU_2/GRU_2/GRUCell/Gates/Linear/Bias" in v.name :
      drug_gru_1_gate_bias.append(v)
   elif "GRU_2/GRU_2/GRUCell/Candidate/Linear/Bias" in v.name :
      drug_gru_1_candidate_bias.append(v)
   elif "GRU_3/GRU_3/GRUCell/Gates/Linear/Matrix" in v.name :
      drug_gru_2_gate_matrix.append(v)
   elif "GRU_3/GRU_3/GRUCell/Candidate/Linear/Matrix" in v.name :
      drug_gru_2_candidate_matrix.append(v)
   elif "GRU_3/GRU_3/GRUCell/Gates/Linear/Bias" in v.name :
      drug_gru_2_gate_bias.append(v)
   elif "GRU_3/GRU_3/GRUCell/Candidate/Linear/Bias" in v.name :
      drug_gru_2_candidate_bias.append(v)
   elif "Embedding_1" in v.name:
      drug_embd_W.append(v)

fc_1 = fully_connected(pool_2, num1_neurons, activation='leakyrelu',weights_init="xavier",name='fully1')
drop_2 = dropout(fc_1, drop_out)
fc_2 = fully_connected(drop_2, num2_neurons, activation='leakyrelu',weights_init="xavier",name='fully2')
drop_3 = dropout(fc_2, drop_out)
linear = fully_connected(drop_3, 5, activation='softmax',name='fully3')
classification = regression(linear, optimizer='adam', learning_rate=learning_rate,
                     loss='categorical_crossentropy', name='target')

# Training
model = tflearn.DNN(classification, tensorboard_verbose=0,tensorboard_dir='./mytensor/',checkpoint_path="./checkpoints/")

#model.load('checkpoints-7500')

#model2 = tflearn.DNN(linear, session = model.session)
######### Setting weights

model.set_weights(prot_gru_1_gate_matrix[0],prot_gru_1_gates_kernel_init)
model.set_weights(prot_gru_1_gate_bias[0],prot_gru_1_gates_bias_init)
model.set_weights(prot_gru_1_candidate_matrix[0],prot_gru_1_candidate_kernel_init)
model.set_weights(prot_gru_1_candidate_bias[0],prot_gru_1_candidate_bias_init)
model.set_weights(prot_gru_2_gate_matrix[0],prot_gru_2_gates_kernel_init)
model.set_weights(prot_gru_2_gate_bias[0],prot_gru_2_gates_bias_init)
model.set_weights(prot_gru_2_candidate_matrix[0],prot_gru_2_candidate_kernel_init)
model.set_weights(prot_gru_2_candidate_bias[0],prot_gru_2_candidate_bias_init)


model.set_weights(drug_gru_1_gate_matrix[0],drug_gru_1_gates_kernel_init)
model.set_weights(drug_gru_1_gate_bias[0],drug_gru_1_gates_bias_init)
model.set_weights(drug_gru_1_candidate_matrix[0],drug_gru_1_candidate_kernel_init)
model.set_weights(drug_gru_1_candidate_bias[0],drug_gru_1_candidate_bias_init)
model.set_weights(drug_gru_2_gate_matrix[0],drug_gru_2_gates_kernel_init)
model.set_weights(drug_gru_2_gate_bias[0],drug_gru_2_gates_bias_init)
model.set_weights(drug_gru_2_candidate_matrix[0],drug_gru_2_candidate_kernel_init)
model.set_weights(drug_gru_2_candidate_bias[0],drug_gru_2_candidate_bias_init)



######## training
model.fit([train_protein,train_compound], {'target': train_IC50}, n_epoch=300,batch_size=64,
           validation_set=([protein_dev,compound_dev], {'target': IC50_dev}),
           snapshot_epoch=True, show_metric=True, run_id='joint_model')

# saving save
model.save('my_model')

'''
print("error on dev")
size = 64
length_dev = len(protein_dev)
print(length_dev)
num_bins = math.ceil(length_dev/size)
for i in range(num_bins):
        if i==0:
          y_pred = model.predict([protein_dev[0:size],compound_dev[0:size]])
        elif i < num_bins-1:
          temp = model.predict([protein_dev[(i*size):((i+1)*size)],compound_dev[(i*size):((i+1)*size)]])
          y_pred = np.concatenate((y_pred,temp), axis=0)
        else:
          temp = model.predict([protein_dev[(i*size):length_dev],compound_dev[(i*size):length_dev]])
          y_pred = np.concatenate((y_pred,temp), axis=0)
row = np.arange(length_dev)
col = np.argmax(y_pred, axis=1)
data = np.ones(length_dev)
y_pred_label = coo_matrix((data, (row, col)), shape=y_pred.shape).toarray()
print(diagnosis(IC50_dev, y_pred_label))
'''
print("error on test0")
size = 64
length_test0 = len(test0_protein)
print(length_test0)
num_bins = math.ceil(length_test0/size)
for i in range(num_bins):
        if i==0:
          y_pred = model.predict([test0_protein[0:size],test0_compound[0:size]])
        elif i < num_bins-1:
          temp = model.predict([test0_protein[(i*size):((i+1)*size)],test0_compound[(i*size):((i+1)*size)]])
          y_pred = np.concatenate((y_pred,temp), axis=0)
        else:
          temp = model.predict([test0_protein[(i*size):length_test0],test0_compound[(i*size):length_test0]])
          y_pred = np.concatenate((y_pred,temp), axis=0)
row = np.arange(length_test0)
col = np.argmax(y_pred, axis=1)
data = np.ones(length_test0)
y_pred_label = coo_matrix((data, (row, col)), shape=y_pred.shape).toarray()
print(diagnosis(test0_IC50, y_pred_label))


print("error on test1")
size = 64
length_test1 = len(test1_protein)
print(length_test1)
num_bins = math.ceil(length_test1/size)
for i in range(num_bins):
        if i==0:
          y_pred = model.predict([test1_protein[0:size],test1_compound[0:size]])
        elif i < num_bins-1:
          temp = model.predict([test1_protein[(i*size):((i+1)*size)],test1_compound[(i*size):((i+1)*size)]])
          y_pred = np.concatenate((y_pred,temp), axis=0)
        else:
          temp = model.predict([test1_protein[(i*size):length_test1],test1_compound[(i*size):length_test1]])
          y_pred = np.concatenate((y_pred,temp), axis=0)
row = np.arange(length_test1)
col = np.argmax(y_pred, axis=1)
data = np.ones(length_test1)
y_pred_label = coo_matrix((data, (row, col)), shape=y_pred.shape).toarray()
print(diagnosis(test1_IC50, y_pred_label))


print("error on test2")
size = 64
length_test2 = len(test2_protein)
print(length_test2)
num_bins = math.ceil(length_test2/size)
for i in range(num_bins):
        if i==0:
          y_pred = model.predict([test2_protein[0:size],test2_compound[0:size]])
        elif i < num_bins-1:
          temp = model.predict([test2_protein[(i*size):((i+1)*size)],test2_compound[(i*size):((i+1)*size)]])
          y_pred = np.concatenate((y_pred,temp), axis=0)
        else:
          temp = model.predict([test2_protein[(i*size):length_test2],test2_compound[(i*size):length_test2]])
          y_pred = np.concatenate((y_pred,temp), axis=0)
row = np.arange(length_test2)
col = np.argmax(y_pred, axis=1)
data = np.ones(length_test2)
y_pred_label = coo_matrix((data, (row, col)), shape=y_pred.shape).toarray()
print(diagnosis(test2_IC50, y_pred_label))

print("error on test3")
size = 64
length_test3 = len(test3_protein)
print(length_test3)
num_bins = math.ceil(length_test3/size)
for i in range(num_bins):
        if i==0:
          y_pred = model.predict([test3_protein[0:size],test3_compound[0:size]])
        elif i < num_bins-1:
          temp = model.predict([test3_protein[(i*size):((i+1)*size)],test3_compound[(i*size):((i+1)*size)]])
          y_pred = np.concatenate((y_pred,temp), axis=0)
        else:
          temp = model.predict([test3_protein[(i*size):length_test3],test3_compound[(i*size):length_test3]])
          y_pred = np.concatenate((y_pred,temp), axis=0)
row = np.arange(length_test3)
col = np.argmax(y_pred, axis=1)
data = np.ones(length_test3)
y_pred_label = coo_matrix((data, (row, col)), shape=y_pred.shape).toarray()
print(diagnosis(test3_IC50, y_pred_label))


print("error on train")
size = 64
length_train = len(train_protein)
print(length_train)
num_bins = math.ceil(length_train/size)
for i in range(num_bins):
        if i==0:
          y_pred = model.predict([train_protein[0:size],train_compound[0:size]])
        elif i < num_bins-1:
          temp = model.predict([train_protein[(i*size):((i+1)*size)],train_compound[(i*size):((i+1)*size)]])
          y_pred = np.concatenate((y_pred,temp), axis=0)
        else:
          temp = model.predict([train_protein[length_train-size:length_train],train_compound[length_train-size:length_train]])
          y_pred = np.concatenate((y_pred,temp[size-length_train+(i*size):size]), axis=0)
row = np.arange(length_train)
col = np.argmax(y_pred, axis=1)
data = np.ones(length_train)
y_pred_label = coo_matrix((data, (row, col)), shape=y_pred.shape).toarray()
print(diagnosis(train_IC50, y_pred_label))

