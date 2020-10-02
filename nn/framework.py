"""
reference:https://github.com/thunlp/Chinese_NRE
modified by wangyan.joy02 on 2019.12.10
"""
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .encoder import BiLstmEncoder
from .classifier import AttClassifier
from torch.autograd import Variable
from torch.nn import functional, init

class MGLattice_model(nn.Module):
    def __init__(self, data):
        super(MGLattice_model, self).__init__()
        # MG-Lattice encoder
        self.encoder = BiLstmEncoder(data)
        # Attentive classifier
        self.classifier = AttClassifier(data)
        self.linear = nn.Linear(in_features=data.HP_hidden_dim*2 , out_features=data.HP_hidden_dim)


    def forward(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, pos1_inputs, pos2_inputs, ins_label, scope, ent1_ids, ent2_ids, marks,omit_types,sent_ids):

        # ins_num * seq_len * hidden_dim
        hidden_out = self.encoder.get_seq_features(gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, pos1_inputs, pos2_inputs,sent_ids)

        zero_vector = torch.zeros((1, 1, hidden_out.size(2)),dtype = torch.float32).cuda()

        if len(marks) == 1:
            if len(ent1_ids) == 1 and len(ent2_ids) == 1:
                ent1_vector = torch.index_select(hidden_out, 1, torch.tensor(ent1_ids[0]).cuda())
                ent2_vector = torch.index_select(hidden_out, 1, torch.tensor(ent2_ids[0]).cuda())
                ent1_vector = torch.mean(ent1_vector, dim=1, keepdim=True)
                ent2_vector = torch.mean(ent2_vector, dim=1, keepdim=True)
                ent_vector = torch.cat((ent1_vector, ent2_vector), dim=2)
                ent_vector = F.relu(self.linear(ent_vector)) #None-linear
                if omit_types[0] !=0:
                    if omit_types[0]==1:
                        hidden_out = torch.cat((hidden_out, ent_vector), dim=1)
                        hidden_out = torch.cat((hidden_out, zero_vector),dim=1)
                        hidden_out = torch.cat((hidden_out, zero_vector), dim=1)

                    elif omit_types[0]==2:
                        hidden_out = torch.cat((hidden_out, zero_vector), dim=1)
                        hidden_out = torch.cat((hidden_out, ent_vector), dim=1)
                        hidden_out = torch.cat((hidden_out, zero_vector), dim=1)

                    elif omit_types[0] == 3:
                        hidden_out = torch.cat((hidden_out, zero_vector), dim=1)
                        hidden_out = torch.cat((hidden_out, zero_vector), dim=1)
                        hidden_out = torch.cat((hidden_out, ent_vector), dim=1)

        elif len(marks) !=1:
            for i in range(0, len(marks)):

                ent1_vector = torch.index_select(hidden_out, 1, torch.tensor(ent1_ids[i]).cuda())
                ent2_vector = torch.index_select(hidden_out, 1, torch.tensor(ent2_ids[i]).cuda())
                ent1_vector = torch.mean(ent1_vector, dim=1, keepdim=True)
                ent2_vector = torch.mean(ent2_vector, dim=1, keepdim=True)

                ent_vector = torch.cat((ent1_vector, ent2_vector), dim=2)
                ent_vector = F.relu(self.linear(ent_vector))  # None-linear

                if omit_types[i] !=0:
                    if omit_types[i] == 1:
                        hidden_out[i] = torch.cat((hidden_out[i], ent_vector), dim=1)
                        hidden_out[i] = torch.cat((hidden_out[i], zero_vector), dim=1)
                        hidden_out[i] = torch.cat((hidden_out[i], zero_vector), dim=1)

                    elif omit_types[i] == 2:
                        hidden_out[i] = torch.cat((hidden_out[i], zero_vector), dim=1)
                        hidden_out[i] = torch.cat((hidden_out[i], ent_vector), dim=1)
                        hidden_out[i] = torch.cat((hidden_out[i], zero_vector), dim=1)

                    elif omit_types[i] == 3:
                        hidden_out[i] = torch.cat((hidden_out[i], zero_vector), dim=1)
                        hidden_out[i] = torch.cat((hidden_out[i], zero_vector), dim=1)
                        hidden_out[i] = torch.cat((hidden_out[i], ent_vector), dim=1)



        logit, alpha = self.classifier.get_logit(hidden_out, ins_label, scope)

        return logit, alpha


