import json

import numpy as np

from collections import Counter

import matplotlib.pyplot as plt


path_en_vocab_freq = 'stat/en_vocab_freq.json'
path_en_len_freq = 'stat/en_len_freq.json'
path_zh_vocab_freq = 'stat/zh_vocab_freq.json'
path_zh_len_freq = 'stat/zh_len_freq.json'

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = ['Arial Unicode MS']


def count(path_freq, items, field):
    pairs = Counter(items)
    sort_items = [item for item, freq in pairs.most_common()]
    sort_freqs = [freq for item, freq in pairs.most_common()]
    item_freq = dict()
    for item, freq in zip(sort_items, sort_freqs):
        item_freq[item] = freq
    with open(path_freq, 'w') as f:
        json.dump(item_freq, f, ensure_ascii=False, indent=4)
    plot_freq(sort_items, sort_freqs, field, u_bound=20)


def plot_freq(items, freqs, field, u_bound):
    inds = np.arange(len(items))
    plt.bar(inds[:u_bound], freqs[:u_bound], width=0.5)
    plt.xlabel(field)
    plt.ylabel('freq')
    plt.xticks(inds[:u_bound], items[:u_bound], rotation='vertical')
    plt.show()


def statistic(path_train):
    with open(path_train, 'r') as f:
        pairs = json.load(f)
    en_texts, zh_texts = zip(*pairs)
    en_texts, zh_texts = list(en_texts), list(zh_texts)
    en_all_words = ' '.join(en_texts).split()
    en_text_lens = [len(text.split()) for text in en_texts]
    zh_text_str = ''.join(zh_texts)
    zh_text_lens = [len(text) for text in zh_texts]
    count(path_en_vocab_freq, en_all_words, 'vocab')
    count(path_en_len_freq, en_text_lens, 'text_len')
    count(path_zh_vocab_freq, zh_text_str, 'vocab')
    count(path_zh_len_freq, zh_text_lens, 'text_len')


if __name__ == '__main__':
    path_train = 'data/train.json'
    statistic(path_train)
