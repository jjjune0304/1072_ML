import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def knn(X_test,X_train,Y_train,k):
    N_train = X_train.shape[0]
    N_test = X_test.shape[0]
    
    predict = np.zeros(N_test,dtype=int)
    distance = cdist(X_test,X_train,'euclidean')
    
    for i in range(N_test):
        d = distance[i,:]
        sorted_index = np.argsort(d)
        neighbor = sorted_index[0:k]
        predict[i] = np.sign(np.sum(Y_train[neighbor]))
    return predict



train, test = np.loadtxt('hw4_train.dat'), np.loadtxt('hw4_test.dat')
N_train, N_test = train.shape[0], test.shape[0]
X_tr, Y_tr = np.array(train[:,:-1]), np.array(train[:,-1],dtype=int)
X_te, Y_te = np.array(test[:,:-1]), np.array(test[:,-1],dtype=int)

'''
problem 11
'''
k = [1,3,5,7,9]
Ein = list()
for i in range(len(k)):
    predict = knn(X_tr,X_tr,Y_tr,k[i])
    Ein.append(float( np.sum(predict != Y_tr) / N_train ))

plt.plot(k,Ein,color='red',label='Ein')
plt.scatter(k,Ein,color='red', s=30)
plt.xticks(k)
plt.xlabel('k')
plt.ylabel('Ein')

for i,j in enumerate(Ein):
    plt.text(k[i]+0.01,j+0.001,'{0:.2f}'.format(j))

plt.savefig('p11.png', dpi=300)
plt.clf()

'''
problem 12
'''
k = [1,3,5,7,9]
Eout = list()
for i in range(len(k)):
    predict = knn(X_te,X_tr,Y_tr,k[i])
    Eout.append(float( np.sum(predict != Y_te) / N_test ))

plt.plot(k,Eout,color='red',label='Ein')
plt.scatter(k,Eout,color='red', s=30)
plt.xticks(k)
plt.xlabel('k')
plt.ylabel('Eout')

for i,j in enumerate(Eout):
    plt.text(k[i]+0.01,j+0.001,'{0:.3f}'.format(j))

plt.savefig('p12.png', dpi=300)
plt.clf()









