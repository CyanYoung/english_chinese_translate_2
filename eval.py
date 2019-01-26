import pickle as pk

import torch

from nltk.translate.bleu_score import corpus_bleu

from build import tensorize

from translate import models, zh_ind_words, predict

from util import map_item


device = torch.device('cpu')

get_text, get_bleu = True, False

path_dev_en_sent = 'feat/en_sent_dev.pkl'
path_dev_zh_sent = 'feat/zh_sent_dev.pkl'
path_dev_label = 'feat/label_dev.pkl'
with open(path_dev_en_sent, 'rb') as f:
    dev_en_sents = pk.load(f)
with open(path_dev_zh_sent, 'rb') as f:
    dev_zh_sents = pk.load(f)
with open(path_dev_label, 'rb') as f:
    dev_labels = pk.load(f)

dev_triples = [dev_en_sents, dev_zh_sents, dev_labels]

path_test_en_sent = 'feat/en_sent_test.pkl'
path_test_label = 'feat/label_test.pkl'
with open(path_test_en_sent, 'rb') as f:
    test_en_sents = pk.load(f)
with open(path_test_label, 'rb') as f:
    test_labels = pk.load(f)


def pair_print(label_mat, prod_mat, ind_words, mode):
    pred_mat = torch.max(prod_mat, dim=-1)[1]
    print('\n%s: \n' % mode)
    for preds, labels in zip(pred_mat, label_mat):
        bound = (labels > 0).sum().item()
        label = ''.join([ind_words[label.item()] for label in labels[:bound]])
        pred = ''.join([ind_words[pred.item()] for pred in preds[:bound]])
        print(' | '.join([label, pred]))


def debug(name, triples, cand):
    en_sents, zh_sents, labels = tensorize(triples, device)
    model = map_item(name, models)
    with torch.no_grad():
        model.eval()
        prods = model(en_sents[:cand], zh_sents[:cand])
    pair_print(labels, prods, zh_ind_words, 'trm')
    encode = map_item(name + '_encode', models)
    decode = map_item(name + '_decode', models)
    with torch.no_grad():
        encode.eval()
        states = encode(en_sents[:cand])
        decode.eval()
        prods = decode(zh_sents[:cand], states)
    pair_print(labels, prods, zh_ind_words, 'trm_sep')


def test(name, test_en_sents, test_labels):
    labels = [[label.split()] for label in test_labels]
    preds = list()
    for en_sent in test_en_sents:
        pred = predict(en_sent, name)
        preds.append(pred.split())
    print('\n%s bleu: %.2f\n' % (name, corpus_bleu(labels, preds)))


if __name__ == '__main__':
    if get_text:
        debug('trm', dev_triples, cand=10)
    if get_bleu:
        test('trm', test_en_sents, test_labels)
