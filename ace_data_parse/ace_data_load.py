# -*- coding: utf-8 -*-
"""
Created on  2020.10.02

@author: wangyan.joy02
"""
import ace_data
import os
import random

#To get relation type
def get_relation_type(res):
    res_all = {}
    for key, value in res.items():
        for sub_key, sub_value in value.items():
            if sub_value[1] in res_all:
                if sub_value[2] not in res_all[sub_value[1]]:
                    res_all[sub_value[1]].append(sub_value[2])
            else:
                res_all.setdefault(sub_value[1],[])
                if sub_value[2] not in res_all[sub_value[1]]:
                    res_all[sub_value[1]].append(sub_value[2])
    return res_all

#To get ACE 2005 Chinese data
def get_relation_data(root_path,relation_type, rel_sub_type):
    # with open(root_path + relation_type + "_" + rel_sub_type + "_relation_data.txt", 'w') as f:
    with open(root_path +  "total.txt", 'a') as f:
        for key, value in res.items():
            for sub_key, sub_value in value.items():
                if sub_value[1] == relation_type and sub_value[2] == rel_sub_type:
                    sentence_in = sub_value[0]+'\t'+sub_value[6][2] + '\t' + sub_value[7][2] + '\t' +relation_type+"/"+rel_sub_type+'\t'+ sub_value[3]
                    print(sentence_in)
                    f.write(sentence_in + '\n')

if __name__ == "__main__":

    root_dir = r"./output2/"
    data = ace_data.load()
    print(data)
    docs = data["docs"]
    nes = data["nes"]
    res = data["res"]
    relation_type = get_relation_type(res)
    # print(relation_type)
    # for key, value in relation_type.items():
    #     for sub_value in  value:
    #         get_relation_data(root_dir, key,  sub_value)
