import pickle

#############################


def load(file_name):
    with open(file_name, 'rb') as f:
        dicti = pickle.load(f)
    return dicti


def save(dicti, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(dicti, f)


def empty(file_name):
    with open(file_name, 'wb'):
        pass


def exists(file_name):
    try:
        f = open(file_name, 'rb')
        f.close()
        return True
    except FileNotFoundError:
        return False


def dump(dicti, file_name):
    with open(file_name, 'ab') as f:
        dicti = pickle.dump(dicti, f)


def load_all(file_name, start=0, end=None):
    with open(file_name, 'rb') as f:
        i = 0
        while end is None or i < end:
            try:
                if start is None or i >= start:
                    yield pickle.load(f)
                else:
                    pickle.load(f)
            except EOFError:
                break
            i += 1
