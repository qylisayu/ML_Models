# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from scipy.special import logsumexp

np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist



#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    inside = -1 * l2(test_datum.T, x_train).T/(2 * (tau ** 2))
    a = np.exp(inside - logsumexp(inside))
    # max = inside.max()
    # a_numerator = np.exp(inside - max)
    # a_denom = a_numerator.sum()
    # a = a_numerator/a_denom
    A = np.diag(a.flatten())
    I = np.identity(len(x_train[0]))
    inverse = np.linalg.solve(x_train.T @ A @ x_train + lam * I, I)
    w = inverse @ x_train.T @ A @ y_train
    w = w.reshape(len(w), 1)
    y_hat = test_datum.T @ w
    return y_hat




def run_validation(x,y,taus,val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''
    valid_loss = []
    num_val = round(val_frac * len(idx))
    id_val = idx[:num_val]
    id_train = idx[num_val:]
    x_train = x[id_train]
    x_val = x[id_val]
    y_train = y[id_train]
    y_val = y[id_val]

    for t in taus:
        y_hat = []
        for i in range(len(x_val)):
            test_datum = x_val[i].reshape(len(x_val[i]), 1)
            y_hat_i = LRLS(test_datum, x_train, y_train, t)
            y_hat.append(y_hat_i[0])
        y_hat = np.array(y_hat)
        y_val = y_val.reshape(len(y_val), 1)
        valid_loss.append((np.square(y_val - y_hat).sum())/len(y_hat))

    valid_loss = np.array(valid_loss)
    return valid_loss


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    #taus = [10]
    taus = np.logspace(1.0,3,200)
    test_losses = run_validation(x,y,taus,val_frac=0.3)
    #print(test_losses)
    #plt.semilogx(train_losses)
    plt.semilogx(taus, test_losses)
    plt.xlabel('Taus')
    plt.ylabel('MSE')
    plt.title('Validation Loss for Various Taus')
    plt.savefig('Validation Loss for Various Taus.jpg')
    plt.show()


