import pickle as pk

from nltk.translate.bleu_score import corpus_bleu

from translate import predict


path_sent1 = 'feat/sent1_test.pkl'
path_label = 'feat/label_test.pkl'
with open(path_sent1, 'rb') as f:
    sent1s = pk.load(f)
with open(path_label, 'rb') as f:
    labels = pk.load(f)


def test(name, sent1s, labels):
    labels = [[label.split()] for label in labels]
    preds = list()
    for sent1 in sent1s:
        pred = predict(sent1, name, 'search')
        preds.append(pred.split())
    print('\n%s bleu: %.2f\n' % (name, corpus_bleu(labels, preds)))


if __name__ == '__main__':
    test('s2s', sent1s, labels)
    test('att', sent1s, labels)
