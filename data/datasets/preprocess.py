# -*- coding: utf-8 -*-

'''
   Read data from JSON files,
   in the meantime, we do preprocess like capitalize the first character of a sentence or normalize digits
'''
import os
import json
from collections import Counter
from nltk.parse import CoreNLPDependencyParser
import numpy as np
import argparse
from tqdm import tqdm
from io_utils import read_yaml
from str_utils import normalize_tok
from vocab import Vocab
from sklearn.model_selection import train_test_split

config = read_yaml('data_config.yaml')
print('seed:', config['random_seed'])
normalize_digits = config['normalize_digits']
lower_case = config['lower_case']

depparser = CoreNLPDependencyParser(url='http://172.28.6.42:9000')


def build_vocab(data_list, file_path):
    token_list = []
    char_list = []
    pos_list = []
    dep_list = []
    for inst in tqdm(data_list, total=len(data_list)):
        words = inst['words']

        temp_parser_res = depparser.parse(words)
        parser_res = []
        for i in temp_parser_res:
            temp = i.to_conll(4).strip().split('\n')
            for t in temp:
                parser_res.append(t.split('\t'))
        pos_list.extend([a[1] for a in parser_res])
        dep_list.extend([a[3] for a in parser_res])

        for word in words:
            word = normalize_tok(word, lower_case, normalize_digits)
            token_list.append(word)
            char_list.extend(list(word))

    token_vocab_file = os.path.join(file_path, config['token_vocab_file'])
    char_vocab_file = os.path.join(file_path, config['char_vocab_file'])
    pos_vocab_file = os.path.join(file_path, config['pos_vocab_file'])
    dep_type_vocab_file = os.path.join(file_path, config['dep_type_vocab_file'])

    print('--------token_vocab---------------')
    token_vocab = Vocab()
    token_vocab.add_spec_toks(unk_tok=True, pad_tok=False)
    token_vocab.add_counter(Counter(token_list))
    token_vocab.save(token_vocab_file)
    print(token_vocab)

    print('--------char_vocab---------------')
    char_vocab = Vocab()
    char_vocab.add_spec_toks(unk_tok=True, pad_tok=False)
    char_vocab.add_counter(Counter(char_list))
    char_vocab.save(char_vocab_file)
    print(char_vocab)

    print('--------pos_vocab---------------')
    pos_vocab = Vocab()
    pos_vocab.add_spec_toks(pad_tok=True, unk_tok=False)
    pos_vocab.add_counter(Counter(pos_list))
    pos_vocab.save(pos_vocab_file)
    print(pos_vocab)

    print('--------dep_vocab---------------')
    dep_vocab = Vocab()
    dep_vocab.add_spec_toks(pad_tok=True, unk_tok=False, self_loop_tok=True)
    dep_vocab.add_counter(Counter(dep_list))
    dep_vocab.save(dep_type_vocab_file)
    print(dep_vocab)

    return dep_vocab, pos_vocab


def load_data_from_ori(file_path):
    pre_id = ''
    data_list = []
    data_dic = {}
    token = []
    opinion = []
    opinion_idx = []
    aspect = []
    aspect_idx = []
    pair = []
    pair_idx = []
    with open(file_path, 'r', encoding='utf-8') as f:
        f.readline()
        while True:
            line = f.readline()
            if line == '':
                token = token_temp
                aspect.append(a_temp)
                aspect_idx.append(a_temp_idx)
                opinion.append(o_temp)
                opinion_idx.append(o_temp_idx)
                pair.extend(pair_temp)
                pair_idx.extend(pair_idx_temp)
                data_dic['words'] = token
                data_dic['aspects'] = aspect
                data_dic['aspects_idx'] = aspect_idx
                data_dic['opinions'] = opinion
                data_dic['opinions_idx'] = opinion_idx
                data_dic['pair'] = pair
                data_dic['pair_idx'] = pair_idx
                data_list.append(data_dic)
                break
            line = line.split('\t')
            if pre_id == line[0]:
                token = token_temp
                aspect.append(a_temp)
                aspect_idx.append(a_temp_idx)
                opinion.append(o_temp)
                opinion_idx.append(o_temp_idx)
                pair.extend(pair_temp)
                pair_idx.extend(pair_idx_temp)
            elif pre_id != '' and pre_id != line[0]:
                token = token_temp
                aspect.append(a_temp)
                aspect_idx.append(a_temp_idx)
                opinion.append(o_temp)
                opinion_idx.append(o_temp_idx)
                pair.extend(pair_temp)
                pair_idx.extend(pair_idx_temp)
                data_dic['words'] = token
                data_dic['aspects'] = aspect
                data_dic['aspects_idx'] = aspect_idx
                data_dic['opinions'] = opinion
                data_dic['opinions_idx'] = opinion_idx
                data_dic['pair'] = pair
                data_dic['pair_idx'] = pair_idx

                data_list.append(data_dic)
                data_dic = {}
                token = []
                opinion = []
                opinion_idx = []
                aspect = []
                aspect_idx = []
                pair = []
                pair_idx = []
            try:
                token_temp = line[1].strip().split()
            except:
                print(line)
            # aspect term
            a_temp = []
            a_temp_idx = []
            for idx, a in enumerate(line[2].strip().split()):
                if a.strip().split('\\')[1] != 'O':
                    a_temp.append(a.strip().split('\\')[0])
                    a_temp_idx.append(idx)
            # opinion term
            o_temp = []
            o_temp_idx = []
            for idx, o in enumerate(line[3].strip().split()):
                if o.strip().split('\\')[1] != 'O':
                    o_temp.append(o.strip().split('\\')[0])
                    o_temp_idx.append(idx)
            pair_temp = []
            pair_idx_temp = []
            pair_temp.append((a_temp, o_temp))
            pair_idx_temp.append((a_temp_idx, o_temp_idx))
            pre_id = line[0]
    return data_list


def sep_discontinuous_term(term_idx):
    """
    according to the source term idx, get the new idx, term rep = [start_idx, end_idx] and it is consequent
    :param term_idx:
    :return:
    """
    bert_idx = []
    flag = 0  # Whether to include discontinuous terms
    for idx in term_idx:
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


def sep_pair_idx(pair_idx):
    """
    according to the source pair idx, get the new idx, term rep = [[start_idx, end_idx], [start_idx, end_idx]]
    and it is consequent
    :param pair_idx:
    :return:
    """
    bert_idx = []
    flag = a_flag = o_flag = 0
    for p in pair_idx:
        bert_a_idx, a_flag = sep_discontinuous_term([p[0]])
        bert_o_idx, o_flag = sep_discontinuous_term([p[1]])
        for a in bert_a_idx:
            for b in bert_o_idx:
                bert_idx.append((a, b))

    if a_flag != 0 or o_flag != 0:
        flag = 1
    return bert_idx, flag


def construct_instance(inst_list, dep_vocab, pos_vocab):
    data = []
    idx = 0
    for inst in tqdm(inst_list, total=len(inst_list)):
        inst_dict = {}
        words = inst['words']
        words_processed = [normalize_tok(w, lower_case, normalize_digits) for w in words]
        temp_parser_res = depparser.parse(words_processed)
        parser_res = []
        for i in temp_parser_res:
            temp = i.to_conll(4).strip().split('\n')
            for t in temp:
                parser_res.append(t.split('\t'))
        words = [a[0] for a in parser_res]
        inst['words'] = words
        inst_dict['tokens'] = words
        aspects_idx = inst['aspects_idx']
        opinions_idx = inst['opinions_idx']
        pair_idx = inst['pair_idx']

        new_aspects_idx, _ = sep_discontinuous_term(aspects_idx)
        new_opinions_idx, _ = sep_discontinuous_term(opinions_idx)
        new_pair_idx, _ = sep_pair_idx(pair_idx)

        s_to_t = {}
        i = j = 0
        while i < len(inst['words']):
            if inst['words'][i] == words[j]:
                s_to_t[i] = [j]
                i += 1
                j += 1
            else:
                s_to_t[i] = []
                if i + 1 > len(inst['words']) - 1:
                    s_to_t[i] = [x for x in range(j, len(words))]
                else:
                    next_token = inst['words'][i + 1]
                    while words[j] != '-RRB-' and words[j] != next_token and words[j] not in next_token and j <= len(
                            words) - 1:
                        s_to_t[i].append(j)
                        j += 1
                i += 1

        def get_new_term(old_term):
            new_term = []
            for i in old_term:
                temp = []
                for j in i:
                    temp.extend(s_to_t[j])
                new_term.append(temp)
            return new_term

        new_aspects = get_new_term(new_aspects_idx)
        new_opinions = get_new_term(new_opinions_idx)
        inst['aspects_idx'] = new_aspects
        inst['opinions_idx'] = new_opinions

        new_pairs = []
        for p in new_pair_idx:
            new_p_a = []
            for a in p[0]:
                new_p_a.extend(s_to_t[a])

            new_p_o = []
            for a in p[1]:
                new_p_o.extend(s_to_t[a])
            new_pairs.append((new_p_a, new_p_o))
        inst['pair_idx'] = new_pairs

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
        inst_dict['dep'] = [a[2] for a in parser_res]
        inst_dict['dep_label'] = [a[3] for a in parser_res]
        inst_dict['dep_label_indices'] = [dep_vocab.get_index(a[3], default_value='dep') for a in parser_res]
        inst_dict['pos'] = [a[1] for a in parser_res]
        inst_dict['pos_indices'] = [pos_vocab.get_index(a[1]) for a in parser_res]
        assert len(inst_dict['tokens']) == len(inst_dict['pos']) == len(inst_dict['dep'])
        # try:
        #     assert len(inst['pos']) == len(inst['pos_indices']) == len(inst['words'])
        # except:
        #     print(inst['pos'])
        #     print(len(inst['pos']))
        #     print(inst['words'])
        #     print(len(inst['words']))
        #     print(len(inst['dep_label']))
        #     print(inst['dep_label'])
        #     print(len(inst['dep']))
        #     idx = inst['words'].index('1/2')
        #     inst_dict['tokens'] = inst['words'][:idx] + inst['words'][idx+1:]
        #     assert len(inst['tag_type']) == len(inst['tag_type_indices']) == len(inst_dict['tokens'])
        #     print(inst_dict['tokens'])
        data.append(inst_dict.copy())
        idx += 1
    return data


def save_data_to_json(train_list, dev_list, test_list, target_dir, dep_vocab, pos_vocab):
    train_path = os.path.join(target_dir, 'train.json')
    dev_path = os.path.join(target_dir, 'dev.json')
    test_path = os.path.join(target_dir, 'test.json')

    train_data = construct_instance(train_list, dep_vocab, pos_vocab)
    dev_data = construct_instance(dev_list, dep_vocab, pos_vocab)
    test_data = construct_instance(test_list, dep_vocab, pos_vocab)

    json.dump(train_data, open(train_path, 'w', encoding='utf-8'))
    json.dump(dev_data, open(dev_path, 'w', encoding='utf-8'))
    json.dump(test_data, open(test_path, 'w', encoding='utf-8'))
    print("preprocess data successful")


def count_men_len_and_rel_dis(inst_lists):
    """
         Count the average length of mention and the distance between the head token of pairs
         Args:
             inst_lists:

         Returns:
            avg_mention_len: float
            avg_rel_distance: float
         """

    mention_lens = []
    rel_distances = []
    for inst in tqdm(inst_lists, total=len(inst_lists)):
        for p in inst["pair_idx"]:
            mention_lens.append(p[0][-1]-p[0][0]+1)
            mention_lens.append(p[1][-1]-p[1][0]+1)
            rel_distances.append(abs(p[0][0]-p[1][0])+1)

        # orl = inst['orl']
        # for x in orl:
        #     if x[-1] == 'DSE':
        #         mention_lens.append(x[3] - x[2] + 1)
        #     elif x[-1] == 'AGENT' or x[-1] == 'TARGET':
        #         mention_lens.append(x[3] - x[2] + 1)
        #         rel_distances.append(abs(x[0]-x[2])+1)
        #     else:
        #         raise KeyError('annotation error, check {}'.format(' '.join(inst['sentences'])))
    avg_mention_len = sum(mention_lens) / len(mention_lens)
    avg_rel_distance = sum(rel_distances) / len(rel_distances)
    print("he average length of mentions: ", avg_mention_len)
    print("the distance between the head token of pairs: ", avg_rel_distance)



if __name__ == '__main__':

    train_list = []
    dev_list = []
    test_list = []
    dataset = ['14lap', '14res', '15res', '16res']
    for d_name in dataset:
        print(d_name)
        file_path = os.path.join(config['ori_data_dir'], d_name)
        train = load_data_from_ori(os.path.join(file_path, 'train.tsv'))
        train, dev = train_test_split(train, test_size=0.2, shuffle=True)
        test = load_data_from_ori(os.path.join(file_path, 'test.tsv'))
        # count_men_len_and_rel_dis(train+dev+test)
        dep_vocab, pos_vocab = build_vocab(train+dev+test, os.path.join(config['new_data_dir'], d_name))
        save_data_to_json(train, dev, test, os.path.join(config['new_data_dir'], d_name),
                          dep_vocab, pos_vocab)

