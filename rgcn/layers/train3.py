from __future__ import print_function

import keras.callbacks
import numpy as np
import pandas as pd
from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from rgcn.layers.graph import GraphConvolution
from rgcn.layers.input_adj import InputAdj
from rgcn.utils import *

import pickle as pkl

import scipy
import numpy.random

import os
import sys
import time
import argparse

np.random.seed()



ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", type=str, default="aifb",
#                 help="Dataset string ('aifb', 'mutag', 'bgs', 'am')")
ap.add_argument("-e", "--epochs", type=int, default=50,
                help="Number training epochs")
ap.add_argument("-hd", "--hidden", type=int, default=64,
                help="Number hidden units")
ap.add_argument("-do", "--dropout", type=float, default=0.,
                help="Dropout rate")
ap.add_argument("-b", "--bases", type=int, default=-1,
                help="Number of bases used (-1: all)")
ap.add_argument("-lr", "--learnrate", type=float, default=0.001,
                help="Learning rate")
ap.add_argument("-l2", "--l2norm", type=float, default=0.,
                help="L2 normalization of input weights")

fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('--validation', dest='validation', action='store_true')
fp.add_argument('--testing', dest='validation', action='store_false')
ap.set_defaults(validation=False)

args = vars(ap.parse_args())
# print(args)

# Define parameters
# DATASET = args['dataset']
NB_EPOCH = args['epochs']
VALIDATION = args['validation']
LR = args['learnrate']
L2 = args['l2norm']
HIDDEN = args['hidden']
BASES = args['bases']
DO = args['dropout']

dirname = os.path.dirname(os.path.realpath(sys.argv[0]))

# with open(dirname + '/' + DATASET + '.pickle', 'rb') as f:
#     data = pkl.load(f)

### changed code starts. rest of the things have been kept same. also no need to run the prepare_dataset

path = r'/home/berry/text_gcn/RGCN/Darkweb'

Adj1 = sp.csr_matrix(np.load(path + '/adjMatrix_creator_msg_post.npy'))
# Adj1 = r[~np.all(r == 0, axis=1)]
Adj2 = sp.csr_matrix(np.load(path + '/adjMatrix_creator_post.npy'))
Adj3 = sp.csr_matrix(np.load(path + '/adjMatrix_direct_friends.npy'))
Adj4 = sp.csr_matrix(np.load(path + '/adjMatrix_direct_notifs.npy'))
Adj5 = sp.csr_matrix(np.load(path + '/adjMatrix_last_reply_msg_posts.npy'))
Adj6 = sp.csr_matrix(np.load(path + '/adjMatrix_last_reply_posts.npy'))

df = pd.read_csv(r'/home/berry/text_gcn/RGCN/Darkweb/orig_scores.csv')
# df = pd.read_csv(r'D:\darkweb\RGCN\Darkweb\orig_scores_multi.csv')


train_idx = df['id'].sample(frac=0.8)
test_idx = df['id'].loc[~df.index.isin(train_idx.index-1)]



# print(test_idx)

# nodes = df.iloc[:,0]
y = sp.csr_matrix(df.iloc[:, -4:])
# number_of_nodes = nodes.shape[0]
# train_idx = train['id']
# test_idx = test['id']
y_train, y_val, y_test, idx_train, idx_val, idx_test = get_splits(y, train_idx,
                                                                  test_idx,
                                                                  VALIDATION)

# A = data['A']
# y = data['y']
# train_idx = data['train_idx']
# test_idx = data['test_idx']

# Get dataset splits
# y_train, y_val, y_test, idx_train, idx_val, idx_test = get_splits(y, train_idx,
#                                                                   test_idx,
#                                                                   VALIDATION)
train_mask = sample_mask(idx_train, y.shape[0])
#
A = [Adj1, Adj2, Adj3, Adj4, Adj5, Adj6]  # changed code: created A

A.append(sp.identity(A[0].shape[0]).tocsr())
num_nodes = A[0].shape[0]
support = len(A)

# Define empty dummy feature matrix (input is ignored as we set featureless=True)
# In case features are available, define them here and set featureless=False.
# X = sp.csr_matrix(A[0].shape)
# X = np.eye(763,763 )
# np.random.shuffle(X)

"""
Initilaising the ALL ZERO rows with Gaussian noise"""

X = np.load(path + '/head_avg_pool.npy')
indices = np.argwhere(np.count_nonzero(X, axis = 1)==0)
for index in indices:
  X[index] = np.random.normal(size=768)
X = sp.csr_matrix(X)



'''
Deleting the All Zero Rows
X = np.load(path + '/head_avg_pool.npy')
indices = np.argwhere(np.count_nonzero(X, axis = 1)==0)
for index in indices:
    updated_X = np.delete(X, index)

 '''


# print(X.shape)
# X = sp.csr_matrix(np.load(path + '/head_avg_pool.npy'))
# head_avg_pool.npy

### changed code ends


# Normalize adjacency matrices individually
for i in range(len(A)):
    d = np.array(A[i].sum(1)).flatten()
    d_inv = 1. / d
    d_inv[np.isinf(d_inv)] = 0.
    D_inv = sp.diags(d_inv)
    A[i] = D_inv.dot(A[i]).tocsr()

A_in = [InputAdj(sparse=True) for _ in range(support)]
X_in = Input(shape=(X.shape[1],), sparse=True)

# Define model architecture
H = GraphConvolution(HIDDEN, support, num_bases=BASES, featureless=False,
                     activation='relu',
                     W_regularizer=l2(L2))([X_in] + A_in)
H = Dropout(DO)(H)
Y = GraphConvolution(y_train.shape[1], support, num_bases=BASES,
                     activation='softmax')([H] + A_in)

# Compile model
model = Model(input=[X_in] + A_in, output=Y)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=LR))

preds = None

# Fit
for epoch in range(1, NB_EPOCH + 1):

    # Log wall-clock time
    t = time.time()

    # Single training iteration
    model.fit([X] + A, y_train, sample_weight=train_mask,
              batch_size=num_nodes, nb_epoch=1, shuffle=False, verbose=0)

    if epoch % 1 == 0:

        # Predict on full dataset
        preds = model.predict([X] + A, batch_size=num_nodes)
        # print(preds)

        # Train / validation scores
        train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
                                                       [idx_train, idx_val])

        print("Epoch: {:04d}".format(epoch),
              "train_loss= {:.4f}".format(train_val_loss[0]),
              "train_acc= {:.4f}".format(train_val_acc[0]),
              "val_loss= {:.4f}".format(train_val_loss[1]),
              "val_acc= {:.4f}".format(train_val_acc[1]),
              "time= {:.4f}".format(time.time() - t))

    else:
        print("Epoch: {:04d}".format(epoch),
              "time= {:.4f}".format(time.time() - t))

# Testing
test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
print("Test set results:",
      "loss= {:.4f}".format(test_loss[0]),
      "accuracy= {:.4f}".format(test_acc[0]))
