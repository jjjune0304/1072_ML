# coding: utf-8
# pylint: disable = invalid-name, C0111
import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

print('Loading data...')
# load or create your dataset

with np.load('X_train.npz') as data:
    X_tr = data['arr_0']
    
with np.load('Y_train.npz') as data:
    Y_tr = data['arr_0']

with np.load('X_test.npz') as data:
    X_te = data['arr_0']

X_tr, X_val, Y_tr, Y_val = train_test_split(X_tr, Y_tr, test_size=0.2,random_state=11)


# Standardize
sc = StandardScaler().fit(X_tr)
X_tr = sc.transform(X_tr)
X_val = sc.transform(X_val)
X_te = sc.transform(X_te)


# create dataset for lightgbm

lgb_train = lgb.Dataset(X_tr[:,0:100], Y_tr[:,2])
lgb_eval = lgb.Dataset(X_val[:,0:100], Y_val[:,2], reference=lgb_train)
# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l1'},
    'num_leaves': 5000,
    'max_depth' : 20,
    'learning_rate': 0.05,
    'verbose': 0,
    'feature_fraction':0.8,
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0
  }
print('Starting training...')
# train
st = time.time()
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=lgb_eval,
                early_stopping_rounds=30)
print('Training is complete. Total time: {:>5.2f}s'.format(time.time()-st))

ax = lgb.plot_importance(gbm, max_num_features=20) #max_features表示最多展示出前10个重要性特征，可以自行设置
plt.show()

# predict
y_tr = gbm.predict(X_tr[:,0:100], num_iteration=gbm.best_iteration)
y_val = gbm.predict(X_val[:,0:100], num_iteration=gbm.best_iteration)
# eval
print("Training set WMAE = {0}".format(WMAE(y_tr,Y_tr[:,2],200)))
print("Validation set WMAE = {0}".format(WMAE(y_val,Y_val[:,2],200)))
print("Training set NAE = {0}".format(NAE(y_tr,Y_tr[:,2])))
print("Validation set NAE = {0}".format(NAE(y_val,Y_val[:,2])))

# test set

y = gbm.predict(X_te[:,0:100], num_iteration=gbm.best_iteration)
np.savetxt('gbm_y2.csv', y, delimiter=',')