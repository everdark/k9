#!/usr/bin/env python
# kylechung 2017-07-20

# experiment with a pure user-itme matrix without additional meta information
# dataset in use is the movielens

#------------------------#
# first lets try lightfm #
#------------------------#
# reference: https://lyst.github.io/lightfm/docs/quickstart.html
# paper: https://arxiv.org/abs/1507.08439
import numpy as np
from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k, auc_score

data = fetch_movielens(min_rating=5.0)
data['item_features'].shape
print(data['item_feature_labels'].shape)
print(data['item_feature_labels'])

print(data['train'].shape)  # 943 users x 1682 items
print(data['test'].shape)  # same dim: same user base with hold-out ratings

latent_dim = 10

model = LightFM(
    loss='warp', 
    no_components=latent_dim, 
    item_alpha=0,  # regularization for item features
    user_alpha=0,  # regularization for user features
    random_state=666)
model.fit(data['train'], epochs=30)

print("Train precision: %.2f" % precision_at_k(model, data['train'], k=5).mean())
print("Test precision: %.2f" % precision_at_k(model, data['test'], k=5).mean())


def sample_recommendation(model, data, user_ids):
    n_users, n_items = data['train'].shape
    for user_id in user_ids:
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        scores = model.predict(user_id, np.arange(n_items))
        top_items = data['item_labels'][np.argsort(-scores)]
        print("User %s" % user_id)
        print("     Known positives:")
        for x in known_positives[:3]:
            print("        %s" % x)
        print("     Recommended:")
        for x in top_items[:3]:
            print("        %s" % x)

sample_recommendation(model, data, [3, 25, 450])

# add regularization
modelr = LightFM(
    loss='warp', 
    no_components=latent_dim, 
    item_alpha=1e-4,
    user_alpha=1e-4,
    random_state=666)
modelr.fit(data['train'], epochs=30)
print("Train precision: %.2f" % precision_at_k(modelr, data['train'], k=5).mean())
print("Test precision: %.2f" % precision_at_k(modelr, data['test'], k=5).mean())


#------------------------------------------------#
# try libmf, a pure matrix factorization library #
#------------------------------------------------#
# https://github.com/cjlin1/libmf
# make the binary avail on the system path for: mf-train, mf-predict
# note that one-class matrix factorization is used and in libmf it is the Bayesian Personalized Ranking model

import os
import sys
import tempfile
import subprocess
import pandas as pd
import scipy as sp
from sklearn.datasets import dump_svmlight_file

tr_file = '/tmp/mf.tr'
te_file = '/tmp/mf.te'
mo_file = '/tmp/mf.model'
pr_file = '/tmp/mf.pred'

# transform input to user-item-score tuple
for f in [tr_file, te_file, mo_file, pr_file]:
    if os.path.exists(f):
        os.remove(f)
for i, j in zip(*data['train'].nonzero()):
    print('%s %s 1' % (i, j), file=open(tr_file, 'a'))
for i, j in zip(*data['test'].nonzero()):
    print('%s %s 1' % (i, j), file=open(te_file, 'a'))


def mf_train(train_file, model_file, learning_rate, k, l2, niter):
    """
    Train a matrix factorization model with libmf.
    The binary 'mf-train' must be callable on the system path.
    """
    p = subprocess.Popen(
        ['mf-train',
         '-r', str(learning_rate),
         '-l2', str(l2),
         '-f', '10',
         '-k', str(k),
         '-t', str(niter),
         train_file, 
         model_file
        ], stdout=sys.stdout)    
    out, err = p.communicate()
    return None

mf_train(tr_file, mo_file, 0.1, latent_dim, 1e-4, 30)

# the resulting model matrix is only 943 X 1656 instead of 943 X 1682 
# because the rest of the items were never rated in the training set
# (this is the side effect of using only 5-star ratings for this dataset)
data['train'].tocsr()[:,1656:]  # 0 stored elements

# lets check the correctness of the model prediction first
# read the solved p q matrix in the model (a little bit hacky cause there is no interface to directly access the matrix)
n_user, n_item = data['train'].shape
model_matrix = pd.read_csv(mo_file, skiprows=5, header=None, sep=' ')
u_matrix = model_matrix.iloc[:n_user,2:2+latent_dim].as_matrix()
i_matrix = model_matrix.iloc[n_user:,2:2+latent_dim].as_matrix()

print(u_matrix.shape)  # the latent factors for users
print(i_matrix.shape)  # the latent factors for items

np.dot(u_matrix[0,], i_matrix[0,].transpose())  # the predicted score for user 0 on item 0


def simple_predict(test_file, model_file, e=10):
    pred_file = tempfile.mktemp()
    p = subprocess.Popen(
        ['mf-predict',
         '-e', str(e),
         test_file, 
         model_file,
         pred_file
        ], stdout=sys.stdout)    
    out, err = p.communicate()
    with open(pred_file, 'r') as f:
        pred = [float(l.strip('\n')) for l in f.readlines()]
    return pred 
    
tmp_file = '/tmp/mf.te.tmp'
print('0 0 1', file=open(tmp_file, 'w'))
s = simple_predict(tmp_file, mo_file)  # should be the same as the dot product above

# lets also verify the MPR metric calculated by libmf
# MPR is: Sigma_ui( r_ui * rank_ui) / Sigma_ui( r_ui )
u0_item_score = np.dot(u_matrix[0,], i_matrix.transpose())  # compute all item score for user 0
u0_item_rank = 1 - sp.stats.rankdata(u0_item_score) / len(u0_item_score)
u0_item_rank[0]  # should be the same as the row-wise MPR above

# one user multiple items
print('0 0 1', file=open(tmp_file, 'w'))
print('0 1 1', file=open(tmp_file, 'a'))
print('0 2 1', file=open(tmp_file, 'a'))
s = simple_predict(tmp_file, mo_file)
sum(u0_item_rank[:3]) / 3

# multiple users and multiple items
print('0 0 1', file=open(tmp_file, 'w'))
print('0 1 1', file=open(tmp_file, 'a'))
print('0 2 1', file=open(tmp_file, 'a'))
print('1 0 1', file=open(tmp_file, 'a'))
print('1 1 1', file=open(tmp_file, 'a'))
s = simple_predict(tmp_file, mo_file)

u1_item_score = np.dot(u_matrix[1,], i_matrix.transpose())
u1_item_rank = 1 - sp.stats.rankdata(u1_item_score) / len(u1_item_score)
np.mean(np.concatenate((u0_item_rank[:3], u1_item_rank[:2])))

# evaluate result for testing set
s = simple_predict(te_file, mo_file)

# re-train with more iterations since loss is not yet converged at 30 rounds
mf_train(tr_file, mo_file, 0.1, latent_dim, 1e-4, 300)
s_tr = simple_predict(tr_file, mo_file)
s_te = simple_predict(te_file, mo_file)

# for item not existed in training?
# notice that the score is always 1
print('0 1656 1', file=open(tmp_file, 'w'))
s = simple_predict(tmp_file, mo_file)


#-----------------------------------------------------#
# compare BPR model implementation: libmf vs. lightfm #
#-----------------------------------------------------#
model = LightFM(
    loss='bpr', 
    no_components=latent_dim, 
    learning_rate=0.1,
    item_alpha=1e-4,
    user_alpha=1e-4,
    random_state=666)
model.fit(data['train'], epochs=300, num_threads=4)

print(auc_score(model, data['train']).mean())  # average over each user
print(auc_score(model, data['test']).mean())
print(auc_score(model, data['test'], data['train']).mean())

# lets first examine the prediction of lightfm
s1 = model.predict(np.array([0]), np.array([0]))
pr = model.predict_rank(data['train']) / data['train'].shape[1]
pr1 = pr[0,0]

# we can resemble the result by manipulate the estimated latent factors
# notice that lightfm implements a generalized factorization: latent factors include both biases and embeddings
u0_item_score_fm = np.dot(model.user_embeddings[0], model.item_embeddings.transpose()) + model.user_biases[0] + model.item_biases
assert np.round(s1, 5) == np.round(u0_item_score_fm[0], 5)
u0_item_rank_fm = 1 - sp.stats.rankdata(u0_item_score_fm) / len(u0_item_score_fm)
assert np.round(pr1, 5) == np.round(u0_item_rank_fm[0], 5)

# lightfm does not implement MPR as evaluation method
# lets calculate it manually:
pr_test = model.predict_rank(data['test']) / data['test'].shape[1]
print('Training MPR: %s' % (sum(pr.data) / len(pr.data)))
print('Testing MPR: %s' % (sum(pr_test.data) / len(pr_test.data)))

pr_test2 = model.predict_rank(data['test'], data['train']) / data['test'].shape[1]  # exclude training set from scoring
print('Testing MPR: %s' % (sum(pr_test2.data) / len(pr_test2.data)))

# it seems that lightfm BPR perform worse than libmf
# try WARP (Weighted Approximate-Rank Pairwise) model instead
warp_model = LightFM(
    loss='warp', 
    no_components=latent_dim, 
    learning_rate=0.1,
    item_alpha=1e-4,
    user_alpha=1e-4,
    random_state=666)
warp_model.fit(data['train'], epochs=300, num_threads=4)

print(auc_score(warp_model, data['train']).mean())
print(auc_score(warp_model, data['test']).mean())
print(auc_score(warp_model, data['test'], data['train']).mean())

pr_test3 = warp_model.predict_rank(data['test'], data['train']) / data['test'].shape[1]
print('Testing MPR: %s' % (sum(pr_test3.data) / len(pr_test3.data)))  # apparently WARP is better than BPR in lightfm
