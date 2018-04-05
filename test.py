#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import h5py
from collections import Counter
import matplotlib.pyplot as plt


def load_data():
	X = []
	Y = []
	with h5py.File("best_100.hdf5", 'r') as f:
		for s in f['samples']:
			data = f['samples/' + s]
			tipo  = data.attrs['type']

			X.append(data[:])
			Y.append(tipo)
	return np.array(X), np.array(Y)
X, Y = load_data()

print(Counter(Y))
# print(X)
# print(Y)
p=[];
for x, y in zip(X, Y):

	if not y in p:
		print('Min y Max')
		print(np.min(x))
		print(np.max(x))
		print('Media')
		print(np.median(x))
		#print(x)
		print(y)
		p.append(y)
		plt.plot(x)
		plt.show()