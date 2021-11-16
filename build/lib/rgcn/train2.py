from __future__ import print_function

import os; os.environ['KERAS_BACKEND'] = 'theano'
import numpy as np
import pandas as pd
from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from rgcn.layers.graph import GraphConvolution
from rgcn.layers.input_adj import InputAdj
from rgcn.utils2 import *
from rgcn.data_utils import *
import random

import pickle as pkl

import scipy
import numpy.random

import os
import sys
import time
import argparse

np.random.seed()

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="ironmarch",
                 help="Dataset string ('aifb', 'mutag', 'bgs', 'am')")
ap.add_argument("-e", "--epochs", type=int, default=500,
                help="Number training epochs")
ap.add_argument("-hd", "--hidden", type=int, default=16,
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
DATASET = args['dataset']
NB_EPOCH = args['epochs']
VALIDATION = args['validation']
LR = args['learnrate']
L2 = args['l2norm']
HIDDEN = args['hidden']
BASES = args['bases']
DO = args['dropout']

dirname = os.path.dirname(os.path.realpath(sys.argv[0]))
print(dirname) # /Users/v_agarwal/Documents/ResearchProblems/darkweb/relational-gcn-master/rgcn


n_class = 3


adjMatrix_filenames = ['adjMatrix_creator_msg_post.npy','adjMatrix_creator_post.npy','adjMatrix_direct_friends.npy','adjMatrix_direct_notifs.npy','adjMatrix_last_reply_msg_posts.npy','adjMatrix_last_reply_posts.npy']

n_relation = len(adjMatrix_filenames)

# load adjacencies from dir
adjacencies = []
for r in range(n_relation):
    A = np.load(dirname + '/data/' + DATASET +'/' + adjMatrix_filenames[r])
    adjacencies.append(sp.csr_matrix(A))
    A_trans = np.transpose(A)
    adjacencies.append(sp.csr_matrix(A_trans))


adjacencies.append(sp.identity(adjacencies[0].shape[0]).tocsr())

# load scores and convert into 1-hot encoding labels
scores = pd.read_csv(dirname + '/data/' + DATASET + '/orig_scores.csv')
y = label_assign(scores["score"], n_class)
n_nodes = y.shape[0]

# dataset splits: Training and Testing


idx = random.sample(range(y.shape[0]), y.shape[0])

train_idx = idx[:int(len(idx)*0.8)]
test_idx = idx[int(len(idx)*0.8):]



y_train, y_val, y_test, idx_train, idx_val, idx_test = get_splits(y, train_idx,
                                                                  test_idx,
                                                                  VALIDATION)

train_mask = sample_mask(idx_train, y.shape[0])

# Define empty dummy feature matrix (input is ignored as we set featureless=True)
# In case features are available, define them here and set featureless=False.
# X = sp.csr_matrix(A[0].shape)
# X = np.eye(763,763 )

X = np.load(dirname + '/data/' + DATASET + '/chunk_avg_pool.npy')

#Initilaising the ALL ZERO rows with Gaussian noise

indices = np.argwhere(np.count_nonzero(X, axis = 1)==0)
for index in indices:
  X[index] = np.random.normal(size=768)

#Converting into sparse format
X = sp.csr_matrix(X)

# np.random.shuffle(X)

#X = sp.csr_matrix(adjacencies[0].shape)

# Normalize adjacency matrices individually
for i in range(len(adjacencies)):
    d = np.array(adjacencies[i].sum(1)).flatten()
    d_inv = 1. / (d+1e-7)
    d_inv[np.isinf(d_inv)] = 0.
    D_inv = sp.diags(d_inv)
    adjacencies[i] = D_inv.dot(adjacencies[i]).tocsr()

support = len(adjacencies)
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
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=LR) )

preds = None

#### added it to change the learning rate according to loss
#change_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience= 5, min_lr= 0.001)
####

# Fit
for epoch in range(1, NB_EPOCH + 1):

    # Log wall-clock time
    t = time.time()

    # Single training iteration
    model.fit([X] + adjacencies, y_train, sample_weight=train_mask,
              batch_size=n_nodes, nb_epoch=1, shuffle=False, verbose=0)

    if epoch % 1 == 0:

        # Predict on full dataset
        preds = model.predict([X] + adjacencies, batch_size=n_nodes)

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
