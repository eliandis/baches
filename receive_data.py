import numpy as np
def load_data_coverted(filename):
    X = []
    Y = []
    f = open(filename, 'r')
    for linea in f.readlines():
        data = linea.replace('\n','').split(' ');
        X.append(data[1:])
        Y.append(data[0])
    f.close()
    return np.array(X), np.array(Y)


