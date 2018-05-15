import numpy as np
import h5py
def load_data():
    X = []
    Y = []
    max_length = 0
    with h5py.File("best_100.hdf5", 'r') as f:
        for s in f['samples']:
            data = f['samples/' + s]
            tipo  = data.attrs['type']
            if max_length < len(data):
                max_length = len(data)
            X.append(data[:])
            Y.append(tipo)
    return np.array(X), np.array(Y), max_length

def padding(x, towrite,max_length, media):
    i = len(towrite)
    ave = round(np.median(x),2)
    while i < max_length:
        if media:
            towrite.append(str(ave))
        else:
            towrite.append('0')
        i=i+1
    return towrite

X, Y, max_length = load_data()
classes = {}
classes['Boyas'] = 1
classes['Bache'] = 2
classes['Bordo'] = 3
classes['Normal'] = 4

nrows= int(X.shape[0]*.7)
trainX = X[0:nrows, ]
trainY = Y[0:nrows, ]
testX = X[nrows+1:, ]
testY = Y[nrows+1:, ]

f = open ("db_text/train_baches_zeros_db.txt", "w")

# Train padding ceros
for x, y in zip(trainX, trainY):
    towrite = []
    towrite.append(str(classes[y]))
    for value in x:
        towrite.append(str(value))
    towrite = padding(x,towrite,max_length+1,0)
    f.write(' '.join(towrite))
    f.write('\n')
f.close()

f = open ("db_text/test_baches_zeros_db.txt", "w")

# Test padding ceros
for x, y in zip(testX, testY):
    towrite = []
    towrite.append(str(classes[y]))
    for value in x:
        towrite.append(str(value))
    towrite = padding(x,towrite,max_length+1,0)
    f.write(' '.join(towrite))
    f.write('\n')
f.close()

f = open ("db_text/train_baches_media_db.txt", "w")
# Train padding media
for x, y in zip(trainX, trainY):
    towrite = []
    towrite.append(str(classes[y]))
    for value in x:
        towrite.append(str(value))
    towrite = padding(x,towrite,max_length+1,1)
    f.write(' '.join(towrite))
    f.write('\n')
f.close()

f = open ("db_text/test_baches_media_db.txt", "w")

# Test padding media
for x, y in zip(testX, testY):
    towrite = []
    towrite.append(str(classes[y]))
    for value in x:
        towrite.append(str(value))
    towrite = padding(x,towrite,max_length+1,1)
    f.write(' '.join(towrite))
    f.write('\n')
f.close()


