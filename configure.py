"""
reference:https://github.com/thunlp/Chinese_NRE
modified by wangyan.joy02 on 2019.12.10
"""

savemodel = "models/ACE.pkl"
loadmodel = "models/ACE.pkl-233"
savedset = "models/ACE.pkl.dset"

public_path = "data"
dataset = "ACE_trigger_full_clean" #ace train and test file
train_file = "train.txt"
dev_file = "valid.txt"
test_file = "test.txt"
relation2id = "relation2id.txt"
char_emb_file = "vec.txt"
sense_emb_file = "sense.txt"
word_sense_map = "sense_map.txt"
max_length = 224

Encoder = 'MGLattice' # 'MGLattice' or 'GRU'
Optimizer = 'Adam' # 'SGD' or 'Adam'
lr = 0.0005 # recommend: 0.015 for SGD ( with lr decay ) and 0.0005 for Adam
weights_mode = 'smooth'

##BERT parameters
bert_floder_path = "data/pretrained_model"
DROPOUT_RATE = 0.3
