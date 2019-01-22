import pickle as pk

import numpy as np

import torch
import torch.nn.functional as F

from preprocess import clean

from represent import sent2ind

from build import get_pos, get_mask

from nn_arch import TrmEncode, TrmDecode

from util import trunc, map_item


def load_model(name, embed_mat, pos_mat, mask_mat, device, mode):
    embed_mat = torch.Tensor(embed_mat)
    model = torch.load(map_item(name, paths), map_location=device)
    full_dict = model.state_dict()
    arch = map_item('_'.join([name, mode]), archs)
    if mode == 'decode':
        part = arch(embed_mat, pos_mat, mask_mat, head, stack).to(device)
    else:
        part = arch(embed_mat, pos_mat, head, stack).to(device)
    part_dict = part.state_dict()
    for key, val in full_dict.items():
        key = trunc(key, num=1)
        if key in part_dict:
            part_dict[key] = val
    part.load_state_dict(part_dict)
    return part


def ind2word(word_inds):
    ind_words = dict()
    for word, ind in word_inds.items():
        ind_words[ind] = word
    return ind_words


def check(probs, cand, keep_eos):
    max_probs, max_inds = list(), list()
    sort_probs = -np.sort(-probs)
    sort_inds = np.argsort(-probs)
    for prob, ind in zip(list(sort_probs), list(sort_inds)):
        if not keep_eos and ind == eos_ind:
            continue
        if ind not in skip_inds:
            max_probs.append(prob)
            max_inds.append(ind)
        if len(max_probs) == cand:
            break
    return max_probs, max_inds


def search(decode, state, cand):
    zh_pad_bos = sent2ind([bos], zh_word_inds, seq_len, keep_oov=True)
    zh_word = torch.LongTensor([zh_pad_bos]).to(device)
    prods = decode(zh_word, state)[0][0]
    probs = F.softmax(prods, dim=0).numpy()
    max_probs, max_inds = check(probs, cand, keep_eos=False)
    zh_texts, log_sums = [bos] * cand, np.log(max_probs)
    fin_zh_texts, fin_logs = list(), list()
    next_words, count = [zh_ind_words[ind] for ind in max_inds], 1
    while cand > 0:
        log_mat, ind_mat = list(), list()
        count = count + 1
        for i in range(cand):
            zh_texts[i] = zh_texts[i] + next_words[i]
            zh_pad_seq = sent2ind(zh_texts[i], zh_word_inds, seq_len, keep_oov=True)
            zh_sent = torch.LongTensor([zh_pad_seq]).to(device)
            step = min(count - 1, seq_len - 1)
            prods = decode(zh_sent, state)[0][step]
            probs = F.softmax(prods, dim=0).numpy()
            max_probs, max_inds = check(probs, cand, keep_eos=True)
            max_logs = np.log(max_probs) + log_sums[i]
            log_mat.append(max_logs)
            ind_mat.append(max_inds)
        max_logs = -np.sort(-np.array(log_mat), axis=None)[:cand]
        next_zh_texts, next_words, log_sums = list(), list(), list()
        for log in max_logs:
            args = np.where(log_mat == log)
            sent_arg, ind_arg = int(args[0][0]), int(args[1][0])
            next_word = zh_ind_words[ind_mat[sent_arg][ind_arg]]
            if next_word != eos and count < max_len:
                next_words.append(next_word)
                next_zh_texts.append(zh_texts[sent_arg])
                log_sums.append(log)
            else:
                cand = cand - 1
                fin_zh_texts.append(zh_texts[sent_arg])
                fin_logs.append(log / count)
        zh_texts = next_zh_texts
    max_arg = np.argmax(np.array(fin_logs))
    return fin_zh_texts[max_arg][1:]


device = torch.device('cpu')

embed_len = 200
seq_len = 50
max_len = 50

head, stack = 4, 2

bos, eos = '<', '>'

pad_ind, oov_ind = 0, 1

path_en_embed = 'feat/en_embed.pkl'
path_en_word_ind = 'feat/en_word_ind.pkl'
path_zh_embed = 'feat/zh_embed.pkl'
path_zh_word_ind = 'feat/zh_word_ind.pkl'
with open(path_en_embed, 'rb') as f:
    en_embed_mat = pk.load(f)
with open(path_en_word_ind, 'rb') as f:
    en_word_inds = pk.load(f)
with open(path_zh_embed, 'rb') as f:
    zh_embed_mat = pk.load(f)
with open(path_zh_word_ind, 'rb') as f:
    zh_word_inds = pk.load(f)

mask_mat = get_mask(head, seq_len).to(device)
pos_mat = get_pos(seq_len, embed_len).to(device)

eos_ind = zh_word_inds[eos]
skip_inds = [pad_ind, oov_ind]

zh_ind_words = ind2word(zh_word_inds)

archs = {'trm_encode': TrmEncode,
         'trm_decode': TrmDecode}

paths = {'trm': 'model/dnn_trm.pkl'}

models = {'trm_encode': load_model('trm', en_embed_mat, pos_mat, mask_mat, device, 'encode'),
          'trm_decode': load_model('trm', zh_embed_mat, pos_mat, mask_mat, device, 'decode')}


def predict(text, name):
    en_text = clean(text, 'en')
    en_words = en_text.split()
    en_pad_seq = sent2ind(en_words, en_word_inds, seq_len, keep_oov=True)
    en_sent = torch.LongTensor([en_pad_seq]).to(device)
    encode = map_item(name + '_encode', models)
    decode = map_item(name + '_decode', models)
    with torch.no_grad():
        encode.eval()
        state = encode(en_sent)
        decode.eval()
        return search(decode, state, cand=3)


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('trm: %s' % predict(text, 'trm'))
