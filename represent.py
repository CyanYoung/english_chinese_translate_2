import json
import pickle as pk

import numpy as np

from gensim.corpora import Dictionary


embed_len = 200
min_freq = 3
en_max_vocab, zh_max_vocab = 10000, 5000
seq_len = 50

bos, eos = '<', '>'

pad_ind, oov_ind = 0, 1

path_en_word_vec = 'feat/en_word_vec.pkl'
path_en_word_ind = 'feat/en_word_ind.pkl'
path_en_embed = 'feat/en_embed.pkl'
path_zh_word_vec = 'feat/zh_word_vec.pkl'
path_zh_word_ind = 'feat/zh_word_ind.pkl'
path_zh_embed = 'feat/zh_embed.pkl'


def load(path):
    with open(path, 'rb') as f:
        item = pk.load(f)
    return item


def save(item, path):
    with open(path, 'wb') as f:
        pk.dump(item, f)


def add_flag(texts):
    flag_texts = list()
    for text in texts:
        flag_texts.append(bos + text + eos)
    return flag_texts


def shift(flag_texts):
    sents = [text[:-1] for text in flag_texts]
    labels = [text[1:] for text in flag_texts]
    return sents, labels


def tran_dict(word_inds, off):
    off_word_inds = dict()
    for word, ind in word_inds.items():
        off_word_inds[word] = ind + off
    return off_word_inds


def tokenize(sent_words, lang, path_word_ind):
    if lang == 'zh':
        sent_words = [list(words) for words in sent_words]
        max_vocab = zh_max_vocab
    else:
        max_vocab = en_max_vocab
    model = Dictionary(sent_words)
    model.filter_extremes(no_below=min_freq, no_above=1.0, keep_n=max_vocab)
    word_inds = model.token2id
    word_inds = tran_dict(word_inds, off=2)
    save(word_inds, path_word_ind)


def embed(path_word_ind, path_word_vec, lang, path_embed):
    word_inds = load(path_word_ind)
    word_vecs = load(path_word_vec)
    if lang == 'zh':
        vocab, max_vocab = word_vecs.vocab, zh_max_vocab
    else:
        vocab, max_vocab = word_vecs.keys(), en_max_vocab
    vocab_num = min(max_vocab + 2, len(word_inds) + 2)
    embed_mat = np.zeros((vocab_num, embed_len))
    for word, ind in word_inds.items():
        if word in vocab:
            if ind < max_vocab:
                embed_mat[ind] = word_vecs[word]
    save(embed_mat, path_embed)


def sent2ind(words, word_inds, seq_len, keep_oov):
    seq = list()
    for word in words:
        if word in word_inds:
            seq.append(word_inds[word])
        elif keep_oov:
            seq.append(oov_ind)
    if len(seq) < seq_len:
        return seq + [pad_ind] * (seq_len - len(seq))
    else:
        return seq[:seq_len]


def align(sent_words, path_word_ind, path_sent):
    word_inds = load(path_word_ind)
    pad_seqs = list()
    for words in sent_words:
        pad_seq = sent2ind(words, word_inds, seq_len, keep_oov=True)
        pad_seqs.append(pad_seq)
    pad_seqs = np.array(pad_seqs)
    save(pad_seqs, path_sent)


def vectorize(paths, mode):
    with open(paths['data'], 'r') as f:
        pairs = json.load(f)
    en_texts, zh_texts = zip(*pairs)
    en_sents, zh_texts = list(en_texts), list(zh_texts)
    en_sent_words = [sent.split() for sent in en_sents]
    flag_zh_texts = add_flag(zh_texts)
    if mode == 'train':
        tokenize(en_sent_words, 'en', path_en_word_ind)
        embed(path_en_word_ind, path_en_word_vec, 'en', path_en_embed)
        tokenize(flag_zh_texts, 'zh', path_zh_word_ind)
        embed(path_zh_word_ind, path_zh_word_vec, 'zh', path_zh_embed)
    if mode == 'test':
        save(en_texts, paths['en_sent'])
        save(zh_texts, paths['label'])
    else:
        zh_sents, labels = shift(flag_zh_texts)
        align(en_sent_words, path_en_word_ind, paths['en_sent'])
        align(zh_sents, path_zh_word_ind, paths['zh_sent'])
        align(labels, path_zh_word_ind, paths['label'])


if __name__ == '__main__':
    paths = dict()
    paths['data'] = 'data/train.json'
    paths['en_sent'] = 'feat/en_sent_train.pkl'
    paths['zh_sent'] = 'feat/zh_sent_train.pkl'
    paths['label'] = 'feat/label_train.pkl'
    vectorize(paths, 'train')
    paths['data'] = 'data/dev.json'
    paths['en_sent'] = 'feat/en_sent_dev.pkl'
    paths['zh_sent'] = 'feat/zh_sent_dev.pkl'
    paths['label'] = 'feat/label_dev.pkl'
    vectorize(paths, 'dev')
    paths['data'] = 'data/test.json'
    paths['en_sent'] = 'feat/en_sent_test.pkl'
    paths['label'] = 'feat/label_test.pkl'
    vectorize(paths, 'test')
