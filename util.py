def map_item(name, items):
    if name in items:
        return items[name]
    else:
        raise KeyError
