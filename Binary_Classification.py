# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 18:03:25 2018

@author: M
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from mpl_toolkits.mplot3d import Axes3D

data2 = scipy.io.loadmat('Dataset\\spam_data.mat')
train_X = data2['train_x']
train_T = data2['train_y']
test_X = data2['test_x']
test_T = data2['test_y']

# Network architecture
input_units = 40
hidden_layer = 3
learning_rate = 0.001
minibatch_size = 20
num_iteration = 400
hidden_units = np.array([10, 5, 3])
train_N = data2['train_x'].shape[0]
test_N = data2['test_x'].shape[0]
batch = train_N/minibatch_size
dot_a = 0.5
plot_N = 100

def softmax(x):
    """Compute the softmax of vector x."""
    exps = np.exp(x)
    return np.divide(exps.T,np.sum(exps,axis=1)).T
    
def cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    """
    log_likelihood = -np.log(X[np.nonzero(y)])
    loss = np.sum(log_likelihood)
    return loss
    
DNN_weight = []
DNN_bias = []
weight_delta = []
bias_delta = []
train_layer_output = []
test_layer_output = []
train_hidden_node = []
test_hidden_node = []
loss_train = []
loss_test = []
train_rate = []
test_rate = []
for l in range(hidden_layer+1):
    if l == 0:
        ni = input_units
    else:
        ni = hidden_units[l-1]
    if l == hidden_layer:
        no = 2
        DNN_bias.append(np.zeros([1,no]))
    else:
        no = hidden_units[l]
        DNN_bias.append(np.zeros([1,no]))
    bound = np.sqrt(6/(ni+no))
    weight_delta.append(np.zeros([ni,no]))
    bias_delta.append(np.zeros([1,no]))
    DNN_weight.append(np.random.uniform(-bound,bound,[ni,no]))
    train_layer_output.append(np.zeros([train_N,no]))
    test_layer_output.append(np.zeros([test_N,no]))
    train_hidden_node.append(np.zeros([train_N,no]))
    test_hidden_node.append(np.zeros([test_N,no]))
    
""" ---training--- """
for it in range(num_iteration):
    """ forward """
    inputs = train_X
    for l in range(hidden_layer+1):
        train_hidden_node[l] = np.dot(inputs,DNN_weight[l])+DNN_bias[l]
        if l == hidden_layer:
            train_layer_output[l] = softmax(train_hidden_node[l])
        else:
            train_layer_output[l] = np.maximum(train_hidden_node[l], 0) # ReLU
        inputs = train_layer_output[l]
    # error function
    loss_train.append(cross_entropy(train_layer_output[3],train_T)/train_N)
    train_rate.append(np.sum(np.argmax(train_T,axis=1)^np.argmax(train_layer_output[3],axis=1))/train_N)
    
    inputs = test_X
    for l in range(hidden_layer+1):
        test_hidden_node[l] = np.dot(inputs,DNN_weight[l])+DNN_bias[l]
        if l == hidden_layer:
            test_layer_output[l] = softmax(test_hidden_node[l])
        else:
            test_layer_output[l] = np.maximum(test_hidden_node[l], 0) # ReLU
        inputs = test_layer_output[l]
    # error function
    loss_test.append(cross_entropy(test_layer_output[3],test_T)/test_N)
    test_rate.append(np.sum(np.argmax(test_T,axis=1)^np.argmax(test_layer_output[3],axis=1))/test_N)
    """ backward """
    rand = np.random.permutation(train_N)
    error = 0
    batch_count = 0
    for n in rand:
        batch_count += 1
        error = train_layer_output[3][n]-train_T[n]
        for l in range(hidden_layer,-1,-1):
            if l == 0:
                weight_delta[l] += np.multiply(train_X[n,None].T,error)
            else:
                weight_delta[l] += np.multiply(train_layer_output[l-1][n,None].T,error)
            bias_delta[l] += error
            if l>0:
                error = np.dot(DNN_weight[l],error.T)*(train_hidden_node[l-1][n]>0)
        if batch_count == minibatch_size:
            batch_count = 0
            for ll in range(hidden_layer+1):
                DNN_weight[ll] -= learning_rate*(weight_delta[ll]/minibatch_size)
                DNN_bias[ll] -= learning_rate*(bias_delta[ll]/minibatch_size)
                weight_delta[ll] = np.zeros(DNN_weight[ll].shape)
                bias_delta[ll] = np.zeros(DNN_bias[ll].shape)
    if it == 9:
        c1 = train_hidden_node[2][np.argmax(train_T,axis=1)==1]
        c2 = train_hidden_node[2][np.argmax(train_T,axis=1)==0]
        if hidden_units[2] == 3:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.set_xlim3d(-5, 5)
            ax.set_ylim3d(-7, 4)
            ax.set_zlim3d(-10, 10)
            plt.plot(c1[0:plot_N,0],c1[0:plot_N,1],c1[0:plot_N,2],'bo')
            plt.plot(c2[0:plot_N,0],c2[0:plot_N,1],c2[0:plot_N,2],'ro')
            plt.title('3D feature 10 epoch')
        else:
            plt.plot(c1[0:plot_N,0],c1[0:plot_N,1],'bo')
            plt.plot(c2[0:plot_N,0],c2[0:plot_N,1],'ro')
            plt.title('2D feature 10 epoch')
        plt.legend(['Class 1','Class 2'])
        plt.show()
    elif it == 389:
        c3 = train_hidden_node[2][np.argmax(train_T,axis=1)==1]
        c4 = train_hidden_node[2][np.argmax(train_T,axis=1)==0]
        if hidden_units[2] == 3:
            fig = plt.figure()
            ax = Axes3D(fig)
            plt.plot(c3[0:plot_N,0],c1[0:plot_N,1],c1[0:plot_N,2],'bo')
            plt.plot(c4[0:plot_N,0],c2[0:plot_N,1],c2[0:plot_N,2],'ro')
            plt.title('3D feature 390 epoch')
        else:
            plt.plot(c3[0:plot_N,0],c3[0:plot_N,1],'bo')
            plt.plot(c4[0:plot_N,0],c4[0:plot_N,1],'ro')
            plt.title('2D feature 390 epoch')
        plt.legend(['Class 1','Class 2'])
        plt.show()

plt.plot(train_rate,'b')
plt.plot(test_rate,'r')
plt.title('error rate')
plt.xlabel('Number of epochs')
plt.ylabel('error rate')
plt.legend(['training','test'])
plt.show()

"""---testing---"""    
inputs = train_X
for l in range(hidden_layer+1):
    train_hidden_node[l] = np.dot(inputs,DNN_weight[l])+DNN_bias[l]
    if l == hidden_layer:
        train_layer_output[l] = softmax(train_hidden_node[l])
    else:
        train_layer_output[l] = train_hidden_node[l]
    inputs = train_layer_output[l]

train_rate = np.sum(np.argmax(train_T,axis=1)^np.argmax(train_layer_output[3],axis=1))/train_N
print('training error rate %.2f%%' % (train_rate*100))

inputs = test_X
for l in range(hidden_layer+1):
    test_hidden_node[l] = np.dot(inputs,DNN_weight[l])+DNN_bias[l]
    if l == hidden_layer:
        test_layer_output[l] = softmax(test_hidden_node[l])
    else:
        test_layer_output[l] = test_hidden_node[l]
    inputs = test_layer_output[l]

test_rate = np.sum(np.argmax(test_T,axis=1)^np.argmax(test_layer_output[3],axis=1))/test_N
print('testing error rate %.2f%%' % (test_rate*100))
    
plt.plot(loss_train,'b')
plt.plot(loss_test,'r')
plt.title('Training loss')
plt.xlabel('Number of epochs')
plt.ylabel('Average cross entropy')
plt.legend(['training loss','test loss'])
plt.show()