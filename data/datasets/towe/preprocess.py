import pickle
import json
import os
import numpy as np
from collections import Counter


def read_pickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def get_bert_term_idx(s_term_idx):
    """
    according to the source term idx, get the new idx, term rep = [start_idx, end_idx] and it is consequent
    :param s_term_idx:
    :return:
    """
    bert_idx = []
    flag = 0  # Whether to include discontinuous terms
    for idx in s_term_idx:
        if len(idx) < 2:
            if [idx[0], idx[-1]] not in bert_idx:
                bert_idx.append([idx[0], idx[-1]])
        else:
            if (idx[0] + len(idx) - 1) == idx[-1]:
                if [idx[0], idx[-1]] not in bert_idx:
                    bert_idx.append([idx[0], idx[-1]])
            else:
                temp_flag = 0
                s_idx = e_idx = idx[0]  # start_idx = end_idx
                for i in idx[1:]:
                    if i == e_idx + 1:
                        e_idx = i
                    else:
                        temp_flag = 1
                        if [s_idx, e_idx] not in bert_idx:
                            bert_idx.append([s_idx, e_idx])
                        s_idx = e_idx = i
                if [s_idx, e_idx] not in bert_idx:
                    bert_idx.append([s_idx, e_idx])

                if temp_flag == 1:
                    flag += 1
                    temp_flag = 0

    return bert_idx, flag


def get_bert_pair_idx(s_pair_idx):
    """
    according to the source pair idx, get the new idx, term rep = [[start_idx, end_idx], [start_idx, end_idx]]
    and it is consequent
    :param s_pair_idx:
    :return:
    """
    bert_idx = []
    flag = a_flag = o_flag = 0
    for p in s_pair_idx:
        bert_a_idx, a_flag = get_bert_term_idx([p[0]])
        bert_o_idx, o_flag = get_bert_term_idx([p[1]])
        for a in bert_a_idx:
            for b in bert_o_idx:
                bert_idx.append((a, b))

    if a_flag != 0 or o_flag != 0:
        flag = 1
    return bert_idx, flag


def construct_instance(inst_list):
    data = []
    for idx, inst in enumerate(inst_list):
        inst_dict = {}
        inst_dict['tokens'] = inst['words']
        # words = inst['words']
        # aspects = inst['aspects']
        aspects_idx = inst['aspects_idx']
        # opinions = inst['opinions']
        opinions_idx = inst['opinions_idx']
        # pair = inst['pair']
        pair_idx = inst['pair_idx']

        new_aspects_idx, _ = get_bert_term_idx(aspects_idx)
        new_opinions_idx, _ = get_bert_term_idx(opinions_idx)
        new_pair_idx, _ = get_bert_pair_idx(pair_idx)
        entities = []
        for a in new_aspects_idx:
            entities.append({'type': "Asp", "start": a[0], "end": a[-1]})
        for o in new_opinions_idx:
            entities.append({'type': "Opi", "start": o[0], "end": o[-1]})
        inst_dict['entities'] = entities.copy()

        relations = []
        for p in new_pair_idx:
            s1, s2 = p[0], p[-1]
            head = new_aspects_idx.index(s1)  # the index of entities
            tail = new_opinions_idx.index(s2)
            relations.append({"type": "Pair", "head": head, "tail": tail+len(new_aspects_idx)})
        inst_dict['relations'] = relations
        inst_dict['orig_id'] = idx
        inst_dict['dep'] = inst['dep']
        inst_dict['dep_label'] = inst['dep_label']
        inst_dict['dep_label_indices'] = inst['dep_label_indices']
        inst_dict['pos'] = inst['tag_type']
        inst_dict['pos_indices'] = inst['tag_type_indices']
        try:
            assert len(inst['tag_type']) == len(inst['tag_type_indices']) == len(inst['words'])
        except:
            print(inst['tag_type'])
            print(len(inst['tag_type']))
            print(inst['words'])
            print(len(inst['words']))
            print(len(inst['dep_label']))
            print(inst['dep_label'])
            print(len(inst['dep']))
            idx = inst['words'].index('1/2')
            inst_dict['tokens'] = inst['words'][:idx] + inst['words'][idx+1:]
            assert len(inst['tag_type']) == len(inst['tag_type_indices']) == len(inst_dict['tokens'])
            print(inst_dict['tokens'])
        data.append(inst_dict.copy())
    return data


def load_data(source_file, target_dir):
    train_list, dev_list, test_list, token_vocab, char_vocab, _, _ = read_pickle(source_file)
    train_path = os.path.join(target_dir, 'train.json')
    dev_path = os.path.join(target_dir, 'dev.json')
    test_path = os.path.join(target_dir, 'test.json')

    train_data = construct_instance(train_list)
    dev_data = construct_instance(dev_list)
    test_data = construct_instance(test_list)

    train = []
    # train = train.(train_data)
    # train = train.append(dev_data)
    train = train_data + dev_data
    print(len(train))
    print(len(train_data))
    print(len(dev_data))
    json.dump(train, open(train_path, 'w', encoding='utf-8'))
    json.dump(dev_data, open(dev_path, 'w', encoding='utf-8'))
    json.dump(test_data, open(test_path, 'w', encoding='utf-8'))
    print("preprocess data successful")


def remove_repeat(terms):
    res = []
    for x in terms:
        if x not in res:
            res.append(x)
    return res


if __name__ == '__main__':
    print('the current file path: ', os.getcwd())
    x = '../../pickle/14lap/data.pl'
    t_dir = './14lap/'
    load_data(x, t_dir)

    # x = '../../pickle/14res/data.pl'
    # t_dir = './towe/14res/'
    # load_data(x, t_dir)
    #
    # x = '../../pickle/15res/data.pl'
    # t_dir = './towe/15res/'
    # load_data(x, t_dir)
    #
    # x = '../../pickle/16res/data.pl'
    # t_dir = './16res/'
    # load_data(x, t_dir)

    