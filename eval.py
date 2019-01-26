import pickle as pk

import torch

from nltk.translate.bleu_score import corpus_bleu

from build import tensorize

from translate import models, zh_ind_words, predict

from util import map_item


device = torch.device('cpu')

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


def pair_print(prod_mat, label_mat, ind_words):
    pred_mat = torch.max(prod_mat, dim=-1)[1]
    for preds, labels in zip(pred_mat, label_mat):
        pairs = list()
        for pred, label in zip(preds, labels):
            pairs.append(' -> '.join([ind_words[pred], ind_words[label]]))
        print(', '.join(pairs))


def check(name, triples, cand):
    en_sents, zh_sents, labels = tensorize(triples, device)
    model = map_item(name, models)
    with torch.no_grad():
        model.eval()
        prods = model(en_sents[:cand], zh_sents[:cand])
    pair_print(prods, labels, zh_ind_words)
    encode = map_item(name + '_encode', models)
    decode = map_item(name + '_decode', models)
    with torch.no_grad():
        encode.eval()
        states = encode(en_sents[:cand])
        decode.eval()
        prods = decode(zh_sents[:cand], states)
    pair_print(prods, labels, zh_ind_words)


def test(name, dev_triples, test_en_sents, test_labels, debug):
    if debug:
        check(name, dev_triples, cand=10)
    labels = [[label.split()] for label in test_labels]
    preds = list()
    for en_sent in test_en_sents:
        pred = predict(en_sent, name)
        preds.append(pred.split())
    print('\n%s bleu: %.2f\n' % (name, corpus_bleu(labels, preds)))


if __name__ == '__main__':
    test('trm', dev_triples, test_en_sents, test_labels, debug=True)
