import sys
import pandas as pd 
import numpy as np
import random
import math
from tqdm import tqdm 

#read csvs
train_X = pd.read_csv(sys.argv[1], header=None).values
train_X = (train_X/255).astype('float32')
train_Y = pd.read_csv(sys.argv[2], header=None).values
test_X = pd.read_csv(sys.argv[3], header=None).values
test_X = (test_X/255).astype('float32')

#one - hot encoding
temp_Y = np.zeros((len(train_Y), 10))
for l in range(len(train_Y)):
	temp_Y[l][train_Y[l][0]] = 1
train_Y = temp_Y

#initialize weights
layers = []
n_input = 784
n_output = 10
n_hidden = int((n_input + n_output)/2)
layer1 = np.empty([n_hidden, n_input])
for i in range(0,n_hidden):
	for j in range(0,n_input):
		layer1[i][j] = random.random()*np.sqrt(1./n_input)
layer2 = np.empty([n_output, n_hidden])
for i in range(0,n_output):
	for j in range(0,n_hidden):
		layer2[i][j] = random.random()*np.sqrt(1./n_hidden)
layers.append(layer1)
layers.append(layer2)
layersB = []
layersB.append(np.zeros((n_hidden, 1)) * np.sqrt(1. / n_input))
layersB.append(np.zeros((n_output, 1)) * np.sqrt(1. / n_hidden))

def forwardProp(data):
	a = np.matmul(layers[0][:, 0:n_input], data.T) + layersB[0]
	hidden_output = 1./(1. + np.exp(-a))
	a = np.matmul(layers[1][:, 0:n_hidden], hidden_output) + layersB[1]
	final_output = np.exp(a) / np.sum(np.exp(a), axis=0)
	return hidden_output, final_output

def backwardProp(hidden_output, final_output, dataY, dataX):
	cost = final_output - dataY.T
	dcost_wo = (1./len(dataY))*np.matmul(cost, hidden_output.T)
	dcost_bo = (1./len(dataY))*np.sum(cost, axis=1, keepdims=True)
	dcost_dah = np.matmul(layers[1][:, 0:n_hidden].T, cost)
	dah_dzh = hidden_output*(1-hidden_output)
	dcost_wh = (1./len(dataY))*np.matmul(dcost_dah*dah_dzh, dataX)
	dcost_bh = (1./len(dataY))*np.sum(dcost_dah*dah_dzh, axis=1, keepdims=True)
	return dcost_bo, dcost_wo, dcost_wh, dcost_bh

def updateWeights(dcost_bo, dcost_wo, dcost_wh, dcost_bh):
	layers[1][:, 0:n_hidden] -= 0.1*dcost_wo
	layersB[1] -= 0.1*dcost_bo
	layers[0][:, 0:n_input] -= 0.1*dcost_wh
	layersB[0] -= 0.1*dcost_bh

for k in range(100):
	batch = 40

	for sample in range(0, len(train_X), batch):
		#forward prop
		hidden_output, final_output = forwardProp(train_X[sample:min(sample+batch, len(train_X))])

		#backward prop
		dcost_bo, dcost_wo, dcost_wh, dcost_bh = backwardProp(hidden_output, final_output, train_Y[sample:min(sample+batch, len(train_X))], train_X[sample:min(sample+batch, len(train_X))])

		#update weights
		updateWeights(dcost_bo, dcost_wo, dcost_wh, dcost_bh)

hidden_output, final_output = forwardProp(test_X)

np.savetxt("test_predictions.csv", np.argmax(final_output, axis=0), fmt ='% s')