#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 00:02:22 2018

@author: ashwin
"""
from __future__ import print_function
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import average_precision_score
import itertools
import pickle
import pdb
from evaluate import evaluateResults as evalRes


class RankSVM(object):
    def __init__(self):
        self.model = LinearSVC()
#        self.featureList = featureList
        
    def train(self,X,y):
        self.model = self.model.fit(X,y)
    
    def test(self,X):
        return self.model.predict(X)
    
    def load_model(self,filename):
        self.model = pickle.load(open(filename,'rb'))
    
    def save_model(self,filename):
        pickle.dump(self.model, open(filename,'wb'))
        
    def rank_docs(self, preds, train_meta, idx):
        if train_meta[idx][1] > 1:
            combinations = list(itertools.permutations(range(train_meta[idx][1]),2))
            rank_list = range(train_meta[idx][1])
            i = 0
            for j,k in combinations:
    #            pdb.set_trace()
                if preds[i] == 1. and rank_list.index(j) > rank_list.index(k):
                    # swap j and k if j > k
                    rank_list[rank_list.index(j)], rank_list[rank_list.index(k)] = rank_list[rank_list.index(k)], rank_list[rank_list.index(j)]
                elif preds[i] == -1. and rank_list.index(j) < rank_list.index(k):
                    # swap j and k if j < k
                    rank_list[rank_list.index(j)], rank_list[rank_list.index(k)] = rank_list[rank_list.index(k)], rank_list[rank_list.index(j)]
                i += 1
        else:
            rank_list = [0]
#        pdb.set_trace()
        return rank_list
            
    

if __name__ == '__main__':
    # rank_svm = RankSVM()
    # full_X = np.load('pairwise_trainX_5b_train.npy')
    # full_y = np.load('pairwise_trainy_5b_train.npy')
    train_meta = pickle.load(open('pairwise_train_snippets_lookup_5b_test.list','rb'))
    
    # data_split = train_meta[int(len(train_meta)*0.8)][-1] # denotes end index
    # meta_split = int(len(train_meta)*0.8)

    # train_X,  train_y = full_X[:data_split+1], full_y[:data_split+1]
    # train_X, train_y = full_X, full_y
    
    # rank_svm.train(train_X, train_y)
    # rank_svm.save_model('rank_svm_5b_train.p')

    rank_svm = RankSVM()
    rank_svm.load_model('rank_svm_5b_train.p')
    rank_list = {}

    # Testing
    full_X = np.load('pairwise_trainX_5b_test.npy')
    # full_y = np.load('pairwise_trainy_5b_test.npy')
    query_ids = pickle.load(open('data/query_ids_5b_test.p','rb'))
    for i in range(0, len(train_meta)): # meta_split+1 if split within your train
        if train_meta[i][1] > 1:
            s_ind = train_meta[i][2]
            e_ind = train_meta[i][3]
            preds = rank_svm.model.predict(full_X[s_ind:e_ind])
            ranked_docs = rank_svm.rank_docs(preds,train_meta,i)
        else:
            ranked_docs = [0]
        rank_list[query_ids[train_meta[i][0]]] = ranked_docs

    pickle.dump(rank_list, open('5btest_rank_results.p','w'))
    #
    # pred_ranks = pickle.load(open('test_rank_results.p'))
    # gold_ranks = pickle.load(open('data/score_gt_unweighted.p'))[meta_split+1:]
    #
    # p1, p3, p5, p10 = evalRes(gold_ranks, pred_ranks)
    # print('P@1 : %s, P@3 : %s, P@5 : %s, P@10 : %s' %(p1,p3,p5,p10))

    
    
    