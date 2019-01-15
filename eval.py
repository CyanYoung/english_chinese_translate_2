import pickle as pk

from nltk.translate.bleu_score import corpus_bleu

from translate import predict


path_en_sent = 'feat/en_sent_test.pkl'
path_label = 'feat/label_test.pkl'
with open(path_en_sent, 'rb') as f:
    en_sents = pk.load(f)
with open(path_label, 'rb') as f:
    labels = pk.load(f)


def test(name, en_sents, labels):
    labels = [[label.split()] for label in labels]
    preds = list()
    for en_sent in en_sents:
        pred = predict(en_sent, name)
        preds.append(pred.split())
    print('\n%s bleu: %.2f\n' % (name, corpus_bleu(labels, preds)))


if __name__ == '__main__':
    test('trm', en_sents, labels)
