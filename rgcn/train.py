from __future__ import print_function

import os; os.environ['KERAS_BACKEND'] = 'theano'
from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
import os
import sys
import time
import argparse
import theano; theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"
import random
import csv
from sklearn.metrics import precision_recall_fscore_support
from rgcn.layers.graph import GraphConvolution
from rgcn.layers.input_adj import InputAdj
from rgcn.utils2 import *
from rgcn.data_utils import *
import keras.backend as K
import numpy

def main(ap):
    #random.seed(10)
    args = vars(ap.parse_args())

    # Define parameters
    DATASET = args['dataset']
    NB_EPOCH = args['epochs']
    VALIDATION = args['validation']
    LR = args['learnrate']
    L2 = args['l2norm']
    HIDDEN = args['hidden']
    HIDDEN2 = args['hidden2']
    BASES = args['bases']
    DO = args['dropout']
    GAUSSIAN = args['gaussian']
    POOL = args['pool']
    config = args['config']
    balanced_loss = args['balanced_loss']
    numHlayers = args['numHlayers']

    adjMatrix_filenames = ['adjMatrix_creator_msg_post.npy', 'adjMatrix_creator_post.npy',
                           'adjMatrix_direct_friends.npy', 'adjMatrix_direct_notifs.npy',
                           'adjMatrix_last_reply_msg_posts.npy', 'adjMatrix_last_reply_posts.npy']
    n_class = 3
    n_relation = len(adjMatrix_filenames)

    # load adjacencies from dir
    dirname = os.path.dirname(os.path.realpath(sys.argv[0]))
    adjacencies = []
    for r in range(n_relation):
        A = np.load(dirname + '/data/' + DATASET + '/' + adjMatrix_filenames[r])
        adjacencies.append(sp.csr_matrix(A))
        A_trans = np.transpose(A)
        adjacencies.append(sp.csr_matrix(A_trans))
    adjacencies.append(sp.identity(adjacencies[0].shape[0]).tocsr())

    # load scores and convert into 1-hot encoding labels
    scores = pd.read_csv(dirname + '/data/' + DATASET + '/orig_scores.csv')
    y = label_assign(scores["score"], n_class)
    true_y = label_assign(scores["score"], n_class).todense()
    n_nodes = y.shape[0]
    label = np.argmax(true_y, 1)

    # dataset splits: Training and Testing
    idx = random.sample(range(y.shape[0]), y.shape[0])
    train_idx = idx[:int(len(idx) * 0.8)]
    test_idx = idx[int(len(idx) * 0.8):]
    y_train, y_val, y_test, idx_train, idx_val, idx_test = get_splits(y, train_idx, test_idx, VALIDATION)
    train_mask = sample_mask(idx_train, y.shape[0])


    featureless = False
    if config == 0:
        X = np.eye(adjacencies[0].shape[0])
        X = sp.csr_matrix(X,adjacencies[0].shape)
    elif config == 1:
        X = np.load(dirname + '/data/' + DATASET + '/chunk_concatenate.npy')
    elif config == 2:
        X = np.load(dirname + '/data/' + DATASET + '/chunk_avg_pool.npy')
    elif config == 3:
        X = np.load(dirname + '/data/' + DATASET + '/head_concatenate.npy')
    elif config == 4:
        X = np.load(dirname + '/data/' + DATASET + '/head_concatenate.npy')
    else:
        print('config should be between 0 and 4.')
    # Initilaising the ALL ZERO rows with Gaussian noise
    if GAUSSIAN == True:
        indices = np.argwhere(np.count_nonzero(X, axis=1) == 0)
        for index in indices:
            X[index] = np.random.normal(size=X.shape[1])

    # Converting into sparse format
    X = sp.csr_matrix(X)

    # Normalize adjacency matrices individually
    for i in range(len(adjacencies)):
        d = np.array(adjacencies[i].sum(1)).flatten()
        d_inv = 1. / (d + 1e-7)
        d_inv[np.isinf(d_inv)] = 0.
        D_inv = sp.diags(d_inv)
        adjacencies[i] = D_inv.dot(adjacencies[i]).tocsr()
    support = len(adjacencies)

    # inputs
    A_in = [InputAdj(sparse=True) for _ in range(support)]
    X_in = Input(shape=(X.shape[1],), sparse=True)

    # Define model architecture
    if numHlayers == 1:
        H1 = GraphConvolution(HIDDEN, support, num_bases=BASES, featureless=featureless,
                              activation='relu', dropout=args["layerDropout"],
                              W_regularizer=l2(L2))([X_in] + A_in)
        H = Dropout(DO)(H1)
        Y = GraphConvolution(y_train.shape[1], support, num_bases=BASES,
                             activation='softmax')([H] + A_in)
    elif numHlayers == 2:
        H1 = GraphConvolution(HIDDEN, support, num_bases=BASES, featureless=featureless,
                              activation='relu', dropout=args["layerDropout"],
                              W_regularizer=l2(L2))([X_in] + A_in)
        H2 = GraphConvolution(HIDDEN2, support, num_bases=BASES, featureless=featureless,
                              activation='relu',
                              W_regularizer=l2(L2))([H1] + A_in)
        H = Dropout(DO)(H2)
        Y = GraphConvolution(y_train.shape[1], support, num_bases=BASES,
                             activation='softmax')([H] + A_in)

    # Compile model
    model = Model(input=[X_in] + A_in, output=Y)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=LR))

    loss = []
    accuracy = []
    # Fit
    for epoch in range(1, NB_EPOCH + 1):

        # Log wall-clock time
        t = time.time()

        # Single training iteration
        model.fit([X] + adjacencies, y_train, sample_weight=train_mask,
                  batch_size=n_nodes, nb_epoch=1, shuffle=False, verbose=0)

        if epoch % 500 == 0:

            # Predict on full dataset
            preds = model.predict([X] + adjacencies, batch_size=n_nodes)

            # Train / validation scores
            train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
                                                           [idx_train, idx_val])

            loss.append(train_val_loss)
            accuracy.append(train_val_acc)

            predLabels = np.argmax(preds, 1)

            train_precision, train_recall, train_f1Score, train_support = precision_recall_fscore_support(
                label[idx_train], predLabels[idx_train], average='weighted')
            val_precision, val_recall, val_f1Score, val_support = precision_recall_fscore_support(
                label[idx_val], predLabels[idx_val], average='weighted')

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
        
    # Testing
    preds = model.predict([X] + adjacencies, batch_size=n_nodes)
    test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
    predLabels = np.argmax(preds, 1)
    test_precision, test_recall, test_f1Score, test_support = precision_recall_fscore_support(label[idx_test],
                                                                                              predLabels[idx_test],
                                                                                              average='weighted')
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
    filename_part = str(HIDDEN) + '_' + str(HIDDEN2) + '_' + str(config) + '.npy'
    np.save(dirname + '/embedding/H1_' + filename_part,out[0])

    F = K.function(inp,[model.layers[15].output])
    out = F([X] + adjacencies)
    print(out[0].shape)
    np.save(dirname + '/embedding/H2_' + filename_part,out[0])

    F = K.function(inp,[model.layers[17].output])
    out = F([X] + adjacencies)
    print(out[0].shape)
    np.save(dirname + '/embedding/out_' + filename_part,out[0])

    np.save(dirname + '/embedding/labels_' + filename_part, label)

    filename_part = str(HIDDEN) + '_' + str(HIDDEN2) + '_' + str(config) + '.csv'
    # ---> print the shape of output from layer 14 which is the first hidden layer in the model
    np.savetxt(dirname + '/results/CSV/train_id_' + filename_part, idx_train, delimiter=",")
    np.savetxt(dirname + '/results/CSV/test_id_' + filename_part, idx_test, delimiter=",")
    np.savetxt(dirname + '/results/CSV/val_id_' + filename_part, idx_val, delimiter=",")

    np.savetxt(dirname + '/results/CSV/loss_' + filename_part, loss, delimiter=",")
    np.savetxt(dirname + '/results/CSV/accuracy_' + filename_part, accuracy, delimiter=",")
    np.savetxt(dirname + '/results/CSV/test_loss_' + filename_part, test_loss, delimiter=",")
    np.savetxt(dirname + '/results/CSV/test_acc_' + filename_part, test_acc, delimiter=",")
    np.savetxt(dirname + '/results/CSV/pred_y_' + filename_part, preds, delimiter=",")
    np.savetxt(dirname + '/results/CSV/true_y_' + filename_part, true_y, delimiter=",")


    return test_loss[0], test_acc[0], test_precision, test_recall, test_f1Score

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", type=str, default="ironmarch",
                    help="Dataset string ('aifb', 'mutag', 'bgs', 'am')")
    ap.add_argument("-e", "--epochs", type=int, default=300,
                    help="Number training epochs")
    ap.add_argument("-hd", "--hidden", type=int, default=64,
                    help="Number hidden units")
    ap.add_argument("-hd2", "--hidden2", type=int, default=128,
                    help="Number hidden units")
    ap.add_argument("-do", "--dropout", type=float, default=0.0,
                    help="Dropout rate")
    ap.add_argument("-b", "--bases", type=int, default=-1,
                    help="Number of bases used (-1: all)")
    ap.add_argument("-lr", "--learnrate", type=float, default=0.0001,
                    help="Learning rate")
    ap.add_argument("-l2", "--l2norm", type=float, default=0.01,
                    help="L2 normalization of input weights")
    ap.add_argument("-GAUSSIAN", "--gaussian", type=bool, default=False,
                    help="Add gaussian noise to X")
    ap.add_argument("-ldo", "--layerDropout", type=float, default=0.1,
                    help="dropout within the GCN layer (for features)")
    ap.add_argument("-POOL", "--pool", type=bool, default=True,
                    help="aggregation technique")
    # new
    ap.add_argument("--config", default=1, type=int, help="configuration")
    ap.add_argument("--numHlayers", default=2, type=int, help="number of hidden layers")
    ap.add_argument("--balanced_loss", default=False, action='store_false')

    fp = ap.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    ap.set_defaults(validation=True)

    numRep = 5
    numHlayers = 1
    ap.set_defaults(numHlayers=numHlayers)

    H = [16, 32, 64, 128]
    for h1 in range(len(H)):
        ap.set_defaults(hidden=H[h1])
        filename = 'results/results_h1_' + str(H[h1]) + '.csv'
        if os.path.exists(filename):
            os.remove(filename)

        configs = [0, 1, 2, 3, 4]
        for config in configs:
            ap.set_defaults(config=config)
            # empty arrays
            acc = np.zeros(numRep)
            precision = np.zeros(numRep)
            recall = np.zeros(numRep)
            F1 = np.zeros(numRep)
            for k in range(numRep):
                a, p, r, f1 = main(ap)  # main code
                acc[k] = a
                precision[k] = p
                recall[k] = r
                F1[k] = f1

            scores = [config, np.mean(acc), np.mean(precision), np.mean(recall), np.mean(F1)]
            with open(filename, 'a+') as fd:
                wr = csv.writer(fd, dialect='excel')
                wr.writerow(scores)
            print('config {} is done.'.format(config))

    numHlayers = 2
    ap.set_defaults(numHlayers=numHlayers)

    #H = [16, 32, 64, 128]
    #H1, H2 = np.meshgrid(H, H)

    H1 = np.array([32])
    H2 = np.array([128])
    for h1 in range(len(H1)):
        #for h2 in range(len(H)):
            ap.set_defaults(hidden=H1[h1])
            ap.set_defaults(hidden2=H2[h1])

            filename = 'results/results_h1_' + str(H1[h1]) + '_h2_' + str(H2[h1]) + '.csv'
            if os.path.exists(filename):
                os.remove(filename)

            configs = [0, 1, 2, 3, 4]
            for config in configs:
                ap.set_defaults(config=config)
                # empty arrays
                acc = []
                precision = []
                recall = []
                F1 = []
                for k in range(numRep):
                    loss, a, p, r, f1 = main(ap)
                    if not np.isinf(loss):
                        acc.append(a)
                        precision.append(p)
                        recall.append(r)
                        F1.append(f1)

                scores = [config, np.mean(acc), np.mean(precision), np.mean(recall), np.mean(F1)]
                with open(filename, 'a+') as fd:
                    wr = csv.writer(fd, dialect='excel')
                    wr.writerow(scores)
                print('config {} is done.'.format(config))
