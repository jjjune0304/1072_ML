import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

def kmeans(data,k):
    N = data.shape[0]
    dim = data.shape[1]
    sample = np.random.randint(0,N,k) # randomly sample k instances
    means = data[sample,:]
    for i in range(200):
        distance = cdist(data,means,'euclidean')
        cluster = np.argmin(distance,axis=1)
        new_means = np.zeros((k,dim),dtype=float)
        for j in range(k):
            member_id = np.where(cluster == j)
            member = data[member_id,:].squeeze()
            # empty cluster
            if(len(member)==0):
                sample = np.random.randint(0,N,1)
                new_means[j] = data[sample,:].squeeze()
                continue
            new_means[j] = np.mean(member,axis=0)
        # check convergence
        if(cdist(means,new_means).all() < 0.001):
            means = new_means
            #print("Convergence at iteration {0}.".format(i+1))
            break
        means = new_means
    # compute Ein
    distance = cdist(data,means,'euclidean')
    cluster = np.argmin(distance,axis=1)
    err = 0.0
    for i in range(N):
        d = np.power(distance[i,cluster[i]],2)
        err = err + d
    err = float(err / N)
    return err

data = np.loadtxt('hw4_nolabel_train.dat')
N = data.shape[0]
k = [2,4,6,8,10]
Ein = np.zeros((len(k),500),dtype=float)
for i in range(5):
    for j in range(500):
        '''
        kmean = KMeans(n_clusters=k[i]).fit(data)
        Ein[i,j] = kmean.inertia_ / N
        '''
        Ein[i,j] = kmeans(data,k[i])

'''
Priblem 15
'''

Ein_avg = np.mean(Ein,axis=1)
print("Ein_avg = {0}".format(Ein_avg))
plt.plot(range(5),Ein_avg,color='red')
plt.scatter(range(5),Ein_avg,color='red', s=30)
plt.xticks(range(5),k)
plt.xlabel('k')
plt.ylabel('average of Ein')

t = range(5)
for i,j in enumerate(Ein_avg):
    plt.text(t[i]+0.01,j+0.001,'{0:.5f}'.format(j))

plt.savefig('p15.png', dpi=300)
plt.clf()

'''
Problem 16
'''
Ein_var = np.var(Ein,axis=1)
print("Ein_var = {0}".format(Ein_var))
plt.plot(range(5),Ein_var,color='red')
plt.scatter(range(5),Ein_var,color='red', s=30)
plt.xticks(range(5),k)
plt.xlabel('k')
plt.ylabel('variance of Ein')

t = range(5)
for i,j in enumerate(Ein_var):
    plt.text(t[i]+0.01,j+0.0001,'{0:.5f}'.format(j))

plt.savefig('p16.png', dpi=300)
plt.clf()









