def load_word(path):
    words = list()
    with open(path, 'r') as f:
        for line in f:
            words.append(line.strip())
    return words


def load_word_re(path):
    words = load_word(path)
    return '(' + ')|('.join(words) + ')'


def trunc(key, num):
    fields = key.split('.')
    return '.'.join(fields[num:])


def map_item(name, items):
    if name in items:
        return items[name]
    else:
        raise KeyError
