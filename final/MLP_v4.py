#!/usr/bin/env python
# coding: utf-8

import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def WMAE(predict,Y,weight):
    N = Y.shape[0]
    diff = np.abs(predict-Y)
    WMAE = np.sum(weight * diff) / N
    return WMAE
    
def NAE(predict,Y):
    N = Y.shape[0]
    diff = np.abs(predict-Y)
    diff = np.divide(diff,Y)
    NAE = np.sum(diff) / N
    return NAE


def err(predict,Y):
    N = Y.shape[0]
    w = [300, 1, 200]
    diff = np.abs(predict-Y)
    
    diff_WMAE = np.multiply(diff,w)
    rate_err, mesh_err, alpha_err = np.sum(diff_WMAE[:,0]) / N, np.sum(diff_WMAE[:,1]) / N, np.sum(diff_WMAE[:,2]) / N
    print('WMAE: penetration rate err = {2}, mesh_size err = {1}, alpha err = {0}'.format(alpha_err,mesh_err,rate_err))
    WMAE = alpha_err + mesh_err + rate_err
    print('WMAE = {0}'.format(WMAE))
    
    diff = np.divide(diff,Y)
    rate_err, mesh_err, alpha_err = np.sum(diff[:,0]) / N, np.sum(diff[:,1]) / N, np.sum(diff[:,2]) / N
    print('NAE: penetration rate err = {2}, mesh_size err = {1}, alpha err = {0}'.format(alpha_err,mesh_err,rate_err))
    NAE = alpha_err + mesh_err + rate_err
    print('NAE = {0}'.format(NAE))



with np.load('X_train.npz') as data:
    X_tr = data['arr_0']
    
with np.load('Y_train.npz') as data:
    Y_tr = data['arr_0']

with np.load('X_test.npz') as data:
    X_te = data['arr_0']
  
X_tr, X_te_val, Y_tr, Y_te_val = train_test_split(X_tr, Y_tr, test_size=0.2,random_state=26)

sc = StandardScaler()
X_tr = sc.fit_transform(X_tr)
X_te_val = sc.transform(X_te_val)
X_te = sc.transform(X_te)

N_train = X_tr.shape[0]
N_test_val = X_te_val.shape[0]
N_test = X_te.shape[0]
D = X_tr.shape[1]


alpha_tr, alpha_te_val, alpha_te = np.zeros(N_train), np.zeros(N_test_val), np.zeros(N_test)
meshsize_tr, meshsize_te_val, meshsize_te = np.zeros(N_train), np.zeros(N_test_val), np.zeros(N_test)
rate_tr, rate_te_val, rate_te = np.zeros(N_train), np.zeros(N_test_val), np.zeros(N_test)

n_iter = 200


random_step = np.random.randint(low=10,high=20,size=n_iter)
st = time.time()
for i in range(n_iter):
    random_id = np.random.choice(N_train, size=int(0.8*N_train))
    selected_features = range(0,D,random_step[i])
    X_train = X_tr[random_id,:]
    X_train = X_train[:,selected_features]
    Y_train = Y_tr[random_id,:]
    # ordinary least squares
    clf = LinearRegression()
    # third target
    clf.fit(X_train, Y_train[:,2])
    alpha_tr += clf.predict(X_tr[:,selected_features])
    alpha_te_val += clf.predict(X_te_val[:,selected_features])
    alpha_te += clf.predict(X_te[:,selected_features])
    #print('iteration {0} is done.'.format(i))

print('Training is complete. Total time: {:>5.2f}s'.format(time.time()-st))
alpha_tr = alpha_tr / n_iter
alpha_te_val = alpha_te_val / n_iter
alpha_te = alpha_te / n_iter

print('train WMAE = {}'.format(WMAE(alpha_tr,Y_tr[:,2],200)))
print('train NAE = {}'.format(NAE(alpha_tr,Y_tr[:,2])))
print('train WMAE = {}'.format(WMAE(alpha_te_val,Y_te_val[:,2],200)))
print('train NAE = {}'.format(NAE(alpha_te_val,Y_te_val[:,2])))

model = MLPRegressor(hidden_layer_sizes=(20,20),activation='relu', random_state=26,
                                       solver='adam',learning_rate='adaptive', verbose=False,
                                       max_iter=10000,learning_rate_init=0.01,alpha=0.01)
st = time.time()
model.fit(X_tr, Y_tr[:,2])
print('Training is complete. Total time: {:>5.2f}s'.format(time.time()-st))

a_tr = model.predict(X_tr)
a_te_val = model.predict(X_te_val)
a_te = model.predict(X_te)

print('train WMAE = {}'.format(WMAE(a_tr,Y_tr[:,2],200)))
print('train NAE = {}'.format(NAE(a_tr,Y_tr[:,2])))
print('train WMAE = {}'.format(WMAE(a_te_val,Y_te_val[:,2],200)))
print('train NAE = {}'.format(NAE(a_te_val,Y_te_val[:,2])))


model = MLPRegressor(hidden_layer_sizes=(16,16,16),activation='relu', random_state=26,
                                       solver='adam',learning_rate='adaptive', verbose=False,
                                       max_iter=10000,learning_rate_init=0.01,alpha=0.01)
st = time.time()
model.fit(X_tr, Y_tr[:,1])
print('Training is complete. Total time: {:>5.2f}s'.format(time.time()-st))

meshsize_tr = model.predict(X_tr)
meshsize_te_val = model.predict(X_te_val)
meshsize_te = model.predict(X_te)

print('train WMAE = {}'.format(WMAE(meshsize_tr,Y_tr[:,1],1)))
print('train NAE = {}'.format(NAE(meshsize_tr,Y_tr[:,1])))
print('train WMAE = {}'.format(WMAE(meshsize_te_val,Y_te_val[:,1],1)))
print('train NAE = {}'.format(NAE(meshsize_te_val,Y_te_val[:,1])))


model = MLPRegressor(hidden_layer_sizes=(40,40),activation='relu', random_state=26,
                                       solver='adam',learning_rate='adaptive',verbose=False,
                                       max_iter=10000,alpha= 0.001)
st = time.time()
y = Y_tr[:,0] * 100
model.fit(X_tr, y)
print('Training is complete. Total time: {:>5.2f}s'.format(time.time()-st))


rate_tr = model.predict(X_tr) / 100
rate_te_val = model.predict(X_te_val) / 100
rate_te = model.predict(X_te) / 100

print('train WMAE = {}'.format(WMAE(rate_tr,Y_tr[:,0],300)))
print('train NAE = {}'.format(NAE(rate_tr,Y_tr[:,0])))
print('train WMAE = {}'.format(WMAE(rate_te_val,Y_te_val[:,0],300)))
print('train NAE = {}'.format(NAE(rate_te_val,Y_te_val[:,0])))



Y_pre_tr = np.array([rate_tr,meshsize_tr,alpha_tr]).T
Y_pre_te_val = np.array([rate_te_val,meshsize_te_val,alpha_te_val]).T
Y_pre_te = np.array([rate_te,meshsize_te,alpha_te]).T

np.savetxt('MLP_test.csv', Y_pre_te, delimiter=',')
err(Y_pre_tr,Y_tr)
err(Y_pre_te_val,Y_te_val)



