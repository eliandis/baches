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
def in_function(X,Y,Y0):
	m = (Y[1]-Y[0])/(X[1]-X[0])
	n = Y[1]-m*X[1]
	if m == 0:
		return [Y0 == Y[1],X[1]]
	else:
		X0 = (Y0 - n)/m
		return  [X[0] <= X0 <= X[1],X0]

def get_frecuency(Y):
	frecuency = []
	period = []
	meansY = []
	meansX = []
	pointsX = []
	pointsY = []
	X = list(range(len(Y)))
	firstX = X[0]
	firstY = Y[0]
	pointsX.append(firstX)
	pointsY.append(firstY)
	count = 0
	i=0
	X0=0
	while i < len(Y)-1:
		res = in_function([i,i+1],[Y[i],Y[i+1]],firstY)
		if res[0]:
			count +=1
		if count == 3:
			T = res[1] - firstX
			f = 1 / T
			firstX = res[1]
			X1 = int(round(firstX))
			temp = Y[X0:X1]
			mean = np.median(temp)
			res_mean = in_function([i, i + 1], [Y[i], Y[i + 1]], mean)
			frecuency.append(f)
			period.append(T)
			meansX.append(res_mean[1])
			meansY.append(mean)
			pointsX.append(firstX)
			pointsY.append(firstY)
			count = 1
			X0 = X1
		i+=1
	return [frecuency,pointsX,pointsY,period,meansX,meansY]


X, Y = load_data()

#print(Counter(Y))
# print(X[0])
# print(Y)
[frecuency,pointsX,pointsY,period,meansX,meansY] = get_frecuency(X[0])
print(len(frecuency))
print(len(pointsX))
print(len(period))
print(len(meansX))
plt.plot(X[0])
plt.plot(pointsX,pointsY,'ro')
plt.plot(meansX,meansY,'bx')
plt.show()
# p=[];
# for x, y in zip(X, Y):
#
# 	if not y in p:
# 		print('Min y Max')
# 		print(np.min(x))
# 		print(np.max(x))
# 		print('Media')
# 		print(np.median(x))
# 		#print(x)
# 		print(y)
# 		p.append(y)
# 		plt.plot(x)
# 		plt.show()