# -*- coding: utf-8 -*-

'''
   Read data from JSON files,
   in the meantime, we do preprocess like capitalize the first character of a sentence or normalize digits
'''
import os

from collections import Counter
from nltk.parse import CoreNLPDependencyParser
import numpy as np
import argparse
from tqdm import tqdm

from io_utils import read_yaml, save_pickle, read_pickle
from str_utils import normalize_tok
from vocab import Vocab
from sklearn.model_selection import train_test_split

config = read_yaml('config.yaml')

parser = argparse.ArgumentParser(description='this is a description')
parser.add_argument('--seed', '-s', required=False, type=int, default=config['random_seed'])
args = parser.parse_args()
config['random_seed'] = args.seed
print('seed:', config['random_seed'])
print('the current file path:', os.getcwd())

np.random.seed(config['random_seed'])

data_dir = config['data_dir']

normalize_digits = config['normalize_digits']
lower_case = config['lower_case']

vocab_dir = config['vocab_dir']


pickle_dir = config['pickle_dir']
vec_npy_file = config['vec_npy']
inst_pl_file = config['inst_pl_file']

lap_14_train_txt = os.path.join(config['lap_14'], 'train.tsv')
lap_14_test_txt = os.path.join(config['lap_14'], 'test.tsv')

res_14_train_txt = os.path.join(config['res_14'], 'train.tsv')
res_14_test_txt = os.path.join(config['res_14'], 'test.tsv')

res_15_train_txt = os.path.join(config['res_15'], 'train.tsv')
res_15_test_txt = os.path.join(config['res_15'], 'test.tsv')

res_16_train_txt = os.path.join(config['res_16'], 'train.tsv')
res_16_test_txt = os.path.join(config['res_16'], 'test.tsv')


POLARITY_DICT = {'NEU': 0, 'POS': 1, 'NEG': 2}
POLARITY_DICT_REV = {v: k for k, v in POLARITY_DICT.items()}

depparser = CoreNLPDependencyParser(url='http://172.28.6.42:9000')


def load_data(txt_path, pair_path):
    """

    :param txt_path: the original annotation file path
    :param pair_path: the processed pair file path
    :return:
    """
    pairs = read_pickle(pair_path)
    data_list = []
    with open(txt_path, encoding='utf-8') as f:
        texts = f.readlines()
    assert len(pairs) == len(texts)
    for idx, (t, p) in enumerate(zip(texts, pairs)):
        data_dic = {}
        temp = t.split('####')
        words = temp[0].split(' ')
        opinion = []
        opinion_idx = []
        aspect = []
        aspect_idx = []
        ps = []
        for i in p:
            a = words[i[0][0]: i[0][-1]+1] if len(i[0]) > 1 else [words[i[0][0]]]
            o = words[i[1][0]: i[1][-1]+1] if len(i[1]) > 1 else [words[i[1][0]]]
            ps.append((a, o, POLARITY_DICT_REV[i[2]]))
            if i[0] not in aspect:
                aspect.append(a)
                aspect_idx.append(i[0])
            if i[1] not in opinion:
                opinion.append(o)
                opinion_idx.append(i[1])

        data_dic['words'] = words
        data_dic['aspects'] = aspect
        data_dic['aspects_idx'] = aspect_idx
        data_dic['opinions'] = opinion
        data_dic['opinions_idx'] = opinion_idx
        data_dic['pair'] = ps
        data_dic['pair_idx'] = p
        data_list.append(data_dic)

    return data_list


def build_vocab(train_list, dev_list, test_list, data_type='lap14'):
    token_list = []
    char_list = []

    aspects_list = []
    opinions_list = []
    pos_list = []
    dep_list = []
    for inst in tqdm(train_list, total=len(train_list)):
        words = inst['words']
        aspects = inst['aspects_idx']  # idx, prds_type
        opinions = inst['opinions_idx']  # arg_id, prd_id, role

        try:
            temp_parser_res = depparser.parse([normalize_tok(w) for w in words])
        except:
            print(words)
            print([normalize_tok(w) for w in words])
            exit(0)
        parser_res = []
        for i in temp_parser_res:
            temp = i.to_conll(4).strip().split('\n')
            for t in temp:
                parser_res.append(t.split('\t'))
        pos_list.extend([a[1] for a in parser_res])
        dep_list.extend([a[3] for a in parser_res])

        for word in words:
            word = normalize_tok(word, lower_case, normalize_digits)
            # if embedd_dict is not None and (word in embedd_dict or word.lower() in embedd_dict):
            token_list.append(word)
            char_list.extend(list(word))

        aspects_list.extend(aspects)
        opinions_list.extend(opinions)

    for inst in tqdm(dev_list, total=len(dev_list)):
        words = inst['words']
        aspects = inst['aspects_idx']  # idx, prds_type
        opinions = inst['opinions_idx']  # arg_id, prd_id, role

        temp_parser_res = depparser.parse([normalize_tok(w) for w in words])
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

        aspects_list.extend(aspects)
        opinions_list.extend(opinions)

    for inst in tqdm(test_list, total=len(test_list)):
        words = inst['words']
        aspects = inst['aspects_idx']  # idx, prds_type
        opinions = inst['opinions_idx']  # arg_id, prd_id, role

        temp_parser_res = depparser.parse([normalize_tok(w) for w in words])

        parser_res = []
        for i in temp_parser_res:
            temp = i.to_conll(4).strip().split('\n')
            for t in temp:
                parser_res.append(t.split('\t'))
        pos_list.extend([a[1] for a in parser_res])
        dep_list.extend([a[3] for a in parser_res])

        for word in words:
            word = normalize_tok(word, lower_case, normalize_digits)
            # if embedd_dict is not None and (word in embedd_dict or word.lower() in embedd_dict):
            token_list.append(word)
            char_list.extend(list(word))

        aspects_list.extend(aspects)
        opinions_list.extend(opinions)

    token_vocab_file = os.path.join(vocab_dir, data_type, config['token_vocab_file'])
    char_vocab_file = os.path.join(vocab_dir, data_type, config['char_vocab_file'])
    pos_vocab_file = os.path.join(vocab_dir, data_type, config['pos_vocab_file'])
    dep_type_vocab_file = os.path.join(vocab_dir, data_type, config['dep_type_vocab_file'])


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
    pos_vocab.add_spec_toks(pad_tok=True, unk_tok=True)
    pos_vocab.add_counter(Counter(pos_list))
    pos_vocab.save(pos_vocab_file)
    print(pos_vocab)

    print('--------dep_vocab---------------')
    dep_vocab = Vocab()
    dep_vocab.add_spec_toks(pad_tok=True, unk_tok=True)
    dep_vocab.add_counter(Counter(dep_list))
    dep_vocab.save(dep_type_vocab_file)
    print(dep_vocab)


def construct_instance(inst_list, token_vocab, char_vocab, pos_vocab, dep_vocab, is_train=True):
    word_num = 0
    processed_inst_list = []
    for inst in tqdm(inst_list, total=len(inst_list)):

        words = inst['words']
        aspects = inst['aspects_idx']
        opinions = inst['opinions_idx']
        pair_idx = inst['pair_idx']

        if is_train and len(pair_idx) == 0:
            continue

        words_processed = []
        word_indices = []
        char_indices = []
        for word in words:
            word = normalize_tok(word, lower_case, normalize_digits)
            words_processed.append(word)
            word_idx = token_vocab.get_index(word)
            word_indices.append(word_idx)
            char_indices.append([char_vocab.get_index(c) for c in word])

        inst['words'] = words_processed
        inst['word_indices'] = word_indices
        inst['char_indices'] = char_indices

        temp_parser_res = depparser.parse(words_processed)
        parser_res = []
        for i in temp_parser_res:
            temp = i.to_conll(4).strip().split('\n')
            for t in temp:
                parser_res.append(t.split('\t'))
        if len(parser_res) > len(inst['words']):
            words = [a[0] for a in parser_res]
            inst['words'] = words
            # print("source text: ", inst['words'])
            # print("new text: ", words)
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
                        while words[j] != '-RRB-' and words[j] != next_token and words[j] not in next_token and j <= len(words) - 1:
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
            new_aspects = get_new_term(aspects)
            new_opinions = get_new_term(opinions)
            inst['aspects_idx'] = new_aspects
            inst['opinions_idx'] = new_opinions

            new_pairs = []
            for p in pair_idx:
                new_p_a = []
                for a in p[0]:
                    new_p_a.extend(s_to_t[a])

                new_p_o = []
                for a in p[1]:
                    new_p_o.extend(s_to_t[a])
                new_pairs.append((new_p_a, new_p_o))
            inst['pair_idx'] = new_pairs

        inst['dep_label'] = [a[3] for a in parser_res]
        inst['dep_label_indices'] = [dep_vocab.get_index(a[3]) for a in parser_res]
        inst['dep'] = [a[2] for a in parser_res]

        inst['tag_type'] = [a[1] for a in parser_res]
        inst['tag_type_indices'] = [pos_vocab.get_index(a[1]) for a in parser_res]

        inst['sent_range'] = list(range(word_num, word_num + len(words)))
        word_num += len(words)
        processed_inst_list.append(inst)

    return processed_inst_list


def pickle_data(train_list, dev_list, test_list, data_type='lap14'):
    token_vocab_file = os.path.join(vocab_dir, data_type, config['token_vocab_file'])
    char_vocab_file = os.path.join(vocab_dir, data_type, config['char_vocab_file'])
    pos_vocab_file = os.path.join(vocab_dir, data_type, config['pos_vocab_file'])
    dep_type_vocab_file = os.path.join(vocab_dir, data_type, config['dep_type_vocab_file'])

    token_vocab = Vocab.load(token_vocab_file)
    char_vocab = Vocab.load(char_vocab_file)
    pos_vocab = Vocab.load(pos_vocab_file)
    dep_vocab = Vocab.load(dep_type_vocab_file)

    processed_train = construct_instance(train_list, token_vocab, char_vocab, pos_vocab, dep_vocab, True)
    processed_dev = construct_instance(dev_list, token_vocab, char_vocab, pos_vocab, dep_vocab, False)
    processed_test = construct_instance(test_list, token_vocab, char_vocab, pos_vocab, dep_vocab, False)

    print('Saving pickle to ', inst_pl_file)
    print('Saving sent size Train: %d, Dev: %d, Test:%d' % (
        len(processed_train), len(processed_dev), len(processed_test)))
    save_pickle(inst_pl_file, [processed_train, processed_dev, processed_test, token_vocab, char_vocab, pos_vocab, dep_vocab])


def load_data_towe(file_path):
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


if __name__ == '__main__':

    # train_list = []
    # dev_list = []
    # test_list = []

    lap_14_train = load_data_towe(lap_14_train_txt)
    lap_14_train, lap_14_dev = train_test_split(lap_14_train, test_size=0.2, shuffle=True)
    lap_14_test = load_data_towe(lap_14_test_txt)
    build_vocab(lap_14_train, lap_14_dev, lap_14_test, 'lap14')
    print('build lap-14 vocab done.')
    pickle_data(lap_14_train, lap_14_dev, lap_14_test, 'lap14')
    print('pickle lap-14 data done.')

    # res_14_train = load_data_towe(res_14_train_txt)
    # res_14_train, res_14_dev = train_test_split(res_14_train, test_size=0.2, shuffle=True)
    # res_14_test = load_data_towe(res_14_test_txt)
    # build_vocab(res_14_train, res_14_dev, res_14_test, 'res14')
    # print('build res-14 vocab done.')
    # pickle_data(res_14_train, res_14_dev, res_14_test, 'res14')
    # print('pickle res-14 data done.')
    #
    # res_15_train = load_data_towe(res_15_train_txt)
    # res_15_train, res_15_dev = train_test_split(res_15_train, test_size=0.2, shuffle=True)
    # res_15_test = load_data_towe(res_15_test_txt)
    # build_vocab(res_15_train, res_15_dev, res_15_test, 'res15')
    # print('build res-15 vocab done.')
    # pickle_data(res_15_train, res_15_dev, res_15_test, 'res15')
    # print('pickle res-15 data done.')
    #
    # res_16_train = load_data_towe(res_16_train_txt)
    # res_16_train, res_16_dev = train_test_split(res_16_train, test_size=0.2, shuffle=True)
    # res_16_test = load_data_towe(res_16_test_txt)
    # build_vocab(res_16_train, res_16_dev, res_16_test, 'res16')
    # print('build res-16 vocab done.')
    # pickle_data(res_16_train, res_16_dev, res_16_test, 'res16')
    # print('pickle res-16 data done.')

    # print(
    #     '14_lap: {}, 14_res: {}, 15_res: {}, 16_res: {}'.format(len(lap_14_train), len(res_14_train), len(res_15_train),
    #                                                             len(res_16_train)))
    # print(
    #     '14_lap: {}, 14_res: {}, 15_res: {}, 16_res: {}'.format(len(lap_14_dev), len(res_14_dev), len(res_15_dev),
    #                                                             len(res_16_dev)))
    # print(
    #     '14_lap: {}, 14_res: {}, 15_res: {}, 16_res: {}'.format(len(lap_14_test), len(res_14_test), len(res_15_test),
    #                                                             len(res_16_test)))