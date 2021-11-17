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
from rgcn.utils import *
from rgcn.data_utils import *
import random

from keras import backend as K

import theano

import pickle as pkl

import scipy
import numpy.random

import os
import sys
import time
import argparse

from sklearn.metrics import precision_recall_fscore_support

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
ap.add_argument("-GAUSSIAN", "--gaussian", type=bool, default=False,
                help="Add gaussian noise to X")
ap.add_argument("-ldo", "--layerDropout", type=float, default=0.,
                help="dropout within the GCN layer (for features)")
ap.add_argument("-POOL", "--pool", type=bool, default=False,
                help="aggregation technique")

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
GAUSSIAN = args['gaussian']
POOL = args['pool']

dirname = os.path.dirname(os.path.realpath(sys.argv[0]))


n_class = 3

# load adjacencies from dir
adjacencies = []
adjMatrix_filenames = ['adjMatrix_creator_msg_post','adjMatrix_creator_post','adjMatrix_direct_friends','adjMatrix_direct_notifs','adjMatrix_last_reply_msg_posts','adjMatrix_last_reply_posts']
n_relation = len(adjMatrix_filenames)

if DATASET=='nulled':
    for r in range(n_relation):
        A = sp.load_npz(dirname + '/data/' + DATASET +'adjMatrix/' + adjMatrix_filenames[r] +'.npz')
        # print(type(A))
        adjacencies.append(A)
        A_trans = sp.csr_matrix.transpose(A)
        adjacencies.append(sp.csr_matrix(A_trans))
else:
    for r in range(n_relation):
        A = np.load(dirname + '/data/' + DATASET +'adjMatrix/' + adjMatrix_filenames[r] + '.npy')
        adjacencies.append(sp.csr_matrix(A))
        A_trans = np.transpose(A)
        adjacencies.append(sp.csr_matrix(A_trans))

adjacencies.append(sp.identity(adjacencies[0].shape[0]).tocsr())

# load scores and convert into 1-hot encoding labels
scores = pd.read_csv(dirname + '/data/' + DATASET + '/scores/orig_scores.csv')
# print(scores["score"])
label = np.floor(scores["score"])
label[label==3] = 2
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

if POOL == False:
    X = np.load(dirname + '/data/' + DATASET + '/nodeFeatures/chunk_concatenate.npy')
else:
    X = np.load(dirname + '/data/' + DATASET + '/nodeFeatures/chunk_avg_pool.npy')
# chunk_concatenate

#Initilaising the ALL ZERO rows with Gaussian noise

if GAUSSIAN==True:
    indices = np.argwhere(np.count_nonzero(X, axis = 1)==0)
    for index in indices:
        X[index] = np.random.normal(size=X.shape[1])

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
H1 = GraphConvolution(HIDDEN, support, num_bases=BASES, featureless=False,
                     activation='relu', dropout= args["layerDropout"],
                     W_regularizer=l2(L2))([X_in] + A_in)
H2 = GraphConvolution(HIDDEN*2, support, num_bases=BASES, featureless=False,
                     activation='relu',
                     W_regularizer=l2(L2))([H1] + A_in)
H3 = Dropout(DO)(H2)
Y = GraphConvolution(y_train.shape[1], support, num_bases=BASES,
                     activation='softmax')([H3] + A_in)

# Compile model
model = Model(input=[X_in] + A_in, output=Y)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=LR) )
# model.summary()
preds = None


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
        predLabels = np.argmax(preds, 1)

        train_precision, train_recall, train_f1Score, train_support = precision_recall_fscore_support(label[idx_train], predLabels[idx_train], average='weighted')
        val_precision, val_recall, val_f1Score, val_support = precision_recall_fscore_support(label[idx_val], predLabels[idx_val], average='weighted')


        print("\n")
        print("Epoch: {:04d}".format(epoch))
        print("Training Stats:",
            "loss= {:.4f}".format(train_val_loss[0]),
            "acc= {:.4f}".format(train_val_acc[0]),
            "p= {:.4f}".format(train_precision),
            "r= {:.4f}".format(train_recall),
            "f1= {:.4f}".format(train_f1Score))
        print("Validation Stats:",
              "loss= {:.4f}".format(train_val_loss[1]),
              "acc= {:.4f}".format(train_val_acc[1]),
              "p= {:.4f}".format(val_precision),
              "r= {:.4f}".format(val_recall),
              "f1= {:.4f}".format(val_f1Score))

    else:
        print("Epoch: {:04d}".format(epoch),
              "time= {:.4f}".format(time.time() - t))

    # if epoch%10==0:
# Testing
test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
test_precision, test_recall, test_f1Score, test_support = precision_recall_fscore_support(label[idx_test], predLabels[idx_test], average='weighted')
print("Test set results:",
              "loss= {:.4f}".format(test_loss[0]),
              "accuracy= {:.4f}".format(test_acc[0]),
              "p= {:.4f}".format(test_precision),
              "r= {:.4f}".format(test_recall),
              "f1= {:.4f}".format(test_f1Score))

inp = model.input
F = K.function(inp,[model.layers[14].output])
out = F([X] + adjacencies)
print(out[0].shape)
np.save('embedding/chunk_concatenate_H1.npy',out[0])

F = K.function(inp,[model.layers[15].output])
out = F([X] + adjacencies)
print(out[0].shape)
np.save('embedding/chunk_concatenate_H2.npy',out[0])

F = K.function(inp,[model.layers[17].output])
out = F([X] + adjacencies)
print(out[0].shape)
np.save('embedding/chunk_concatenate_out.npy',out[0])

np.save("labels.npy", label)

# ---> print the shape of output from layer 14 which is the first hidden layer in the model
