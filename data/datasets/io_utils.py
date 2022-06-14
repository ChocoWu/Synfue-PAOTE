# -*- coding: utf-8 -*-

import os
import numpy as np

import pickle
import json
import yaml
import sys
import logging


def read_json_lines(path):
    res = []
    for line in open(path, 'r'):
        res.append(json.loads(line))
    return res


def read_yaml(path, encoding='utf-8'):
    with open(path, 'r', encoding=encoding) as file:
        return yaml.load(file.read())


def read_lines(path, encoding='utf-8', return_list=False):
    with open(path, 'r', encoding=encoding) as file:
        if return_list:
            return file.readlines()
        for line in file:
            yield line.strip()


def read_multi_line_sent(path, skip_line_start='-DOCSTART-'):
    sent_lines = []
    for line in read_lines(path):
        line = line.strip()
        if len(line) == 0 or line.startswith(skip_line_start):
            if sent_lines:
                yield sent_lines
                sent_lines = []
            continue
        sent_lines.append(line)
    if sent_lines:
        yield sent_lines


def read_pickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def save_pickle(path, obj):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)


def write_lines(txt_path, lines):
    with open(txt_path, 'w', encoding='utf-8') as file:
        for line in lines:
            file.write(line + '\n')


def write_texts(txt_path, lines):
    with open(txt_path, 'w', encoding='utf-8') as file:
        for line in lines:
            file.write(line)


def load_embed_txt(embed_file):
    """Load embed_file into a python dictionary.
    Note: the embed_file should be a Glove formated txt file. Assuming
    embed_size=5, for example:
    the -0.071549 0.093459 0.023738 -0.090339 0.056123
    to 0.57346 0.5417 -0.23477 -0.3624 0.4037
    and 0.20327 0.47348 0.050877 0.002103 0.060547
    Args:
    embed_file: file path to the embedding file.
    Returns:
    a dictionary that maps word to vector, and the size of embedding dimensions.
    """
    emb_dict = dict()
    emb_size = None
    for line in read_lines(embed_file):
      tokens = line.strip().split(" ")
      word = tokens[0]
      vec = list(map(float, tokens[1:]))
      emb_dict[word] = vec
      if emb_size:
        assert emb_size == len(vec), "All embedding size should be same."
      else:
        emb_size = len(vec)
    return emb_dict, emb_size


def txt_to_npy(dirname, fname, output_name):
    emb_dict, emb_size = load_embed_txt(os.path.join(dirname,fname))
    words = []
    vec = np.empty((len(emb_dict), emb_size))
    i = 0
    for word, vec_list in emb_dict.items():
        words.append(word)
        vec[i] = vec_list
        i += 1
    with open(os.path.join(dirname,output_name+'.vocab.txt'), 'w') as file:
        for word in words:
            file.write(word+'\n')
    np.save(os.path.join(dirname,output_name+'.npy'), vec)


def get_logger(name, log_dir=None, log_name=None, file_model='a',
               level=logging.INFO, handler=sys.stdout,
               formatter='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(formatter)
    stream_handler = logging.StreamHandler(handler)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_dir and log_name:
        filename = os.path.join(log_dir, log_name)
        file_handler = logging.FileHandler(filename, encoding='utf-8', mode=file_model)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def trigger_tag_to_list(trigger_tags, O_idx):
    trigger_list = []
    for i, trigger_type in enumerate(trigger_tags):
        if trigger_type != O_idx:
            trigger_list.append([i, trigger_type])
    return trigger_list


def relative_position(ent_start, ent_end, tok_idx, max_position_len=150):
    if ent_start <= tok_idx <= ent_end:
        return 0
    elif tok_idx < ent_start:
        return ent_start - tok_idx
    elif tok_idx > ent_end:
        return tok_idx - ent_end + max_position_len

    return None


def to_set(*list_vars):
    sets = []
    for list_var in list_vars:
        sets.append({tuple(sub_list) for sub_list in list_var})
    return sets