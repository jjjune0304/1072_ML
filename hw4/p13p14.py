import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def uniform_rbf(X_test,X_train,Y_train,gamma):
    
    distance = -1 * gamma * np.power(cdist(X_test,X_train,'euclidean'),2)
    value = np.exp(distance).dot(Y_train)
    predict = np.sign(value)
    return predict


train, test = np.loadtxt('hw4_train.dat'), np.loadtxt('hw4_test.dat')
N_train, N_test = train.shape[0], test.shape[0]
X_tr, Y_tr = np.array(train[:,:-1]), np.array(train[:,-1],dtype=int)
X_te, Y_te = np.array(test[:,:-1]), np.array(test[:,-1],dtype=int)

'''
problem 13
'''
gamma = [0.001,0.1,1,10,100]
Ein = list()
for i in range(len(gamma)):
    predict = uniform_rbf(X_tr,X_tr,Y_tr,gamma[i])
    Ein.append(float( np.sum(predict != Y_tr) / N_train ))

plt.plot(range(5),Ein,color='red',label='Ein')
plt.scatter(range(5),Ein,color='red', s=30)
plt.xticks(range(5),gamma)
plt.xlabel('gamma')
plt.ylabel('Ein')

t = range(5)
for i,j in enumerate(Ein):
    plt.text(t[i]+0.01,j+0.001,'{0:.2f}'.format(j))

plt.savefig('p13.png', dpi=300)
plt.clf()

'''
problem 14
'''
gamma = [0.001,0.1,1,10,100]
Eout = list()
for i in range(len(gamma)):
    predict = uniform_rbf(X_te,X_tr,Y_tr,gamma[i])
    Eout.append(float( np.sum(predict != Y_te) / N_test ))

plt.plot(range(5),Eout,color='red')
plt.scatter(range(5),Eout,color='red', s=30)
plt.xticks(range(5),gamma)
plt.xlabel('gamma')
plt.ylabel('Eout')

t = range(5)
for i,j in enumerate(Eout):
    plt.text(t[i]+0.01,j+0.001,'{0:.3f}'.format(j))

plt.savefig('p14.png', dpi=300)
plt.clf()




















