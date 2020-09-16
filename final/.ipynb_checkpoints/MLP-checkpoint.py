{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.neural_network import MLPRegressor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def err(predict,Y):\n",
    "    N = Y.shape[0]\n",
    "    w = [200, 1, 300]\n",
    "    diff = np.abs(predict-Y)\n",
    "    \n",
    "    diff_WMAE = np.multiply(diff,w)\n",
    "    alpha_err, mesh_err, rate_err = np.sum(diff_WMAE[:,0]) / N, np.sum(diff_WMAE[:,1]) / N, np.sum(diff_WMAE[:,2]) / N\n",
    "    print('WMAE: alpha err = {0}, mesh_size err = {1}, penetration rate err = {2}'.format(alpha_err,mesh_err,rate_err))\n",
    "    WMAE = alpha_err + mesh_err + rate_err\n",
    "    print('WMAE = {0}'.format(WMAE))\n",
    "    \n",
    "    diff = np.divide(diff,Y)\n",
    "    alpha_err, mesh_err, rate_err = np.sum(diff[:,0]) / N, np.sum(diff[:,1]) / N, np.sum(diff[:,2]) / N\n",
    "    print('NAE: alpha err = {0}, mesh_size err = {1}, penetration rate err = {2}'.format(alpha_err,mesh_err,rate_err))\n",
    "    NAE = alpha_err + mesh_err + rate_err\n",
    "    print('NAE = {0}'.format(NAE))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "with np.load('X_train.npz') as data:\n",
    "    X_tr = data['arr_0']\n",
    "    \n",
    "with np.load('Y_train.npz') as data:\n",
    "    Y_tr = data['arr_0']\n",
    "\n",
    "with np.load('X_test.npz') as data:\n",
    "    X_te = data['arr_0']\n",
    "  \n",
    "X_tr, X_te_val, Y_tr, Y_te_val = train_test_split(X_tr, Y_tr, test_size=0.33, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "N_train = X_tr.shape[0]\n",
    "N_test_val = X_te_val.shape[0]\n",
    "N_test = X_te.shape[0]\n",
    "D = X_tr.shape[1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = MLPRegressor(hidden_layer_sizes=(5,),activation='relu',\n",
    "                                       solver='adam',learning_rate='adaptive',\n",
    "                                       max_iter=1000,learning_rate_init=0.01,alpha=0.01)\n",
    "\n",
    "model.fit(X_tr, Y_tr)\n",
    "Y_pre_tr = predict(X_tr)\n",
    "Y_pre_te_val = predict(X_te_val)\n",
    "Y_pre_te = predict(X_te)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
