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
import pickle
import pdb


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
        
    def rank_docs(self, preds):
        
    

if __name__ == '__main__':
    rank_svm = RankSVM()
    full_X = np.load('pairwise_trainX_6b.npy')
    full_y = np.load('pairwise_trainy_6b.npy')
    train_meta = pickle.load(open('train_snippets_meta.list','rb'))
    
    data_split = train_meta[int(len(train_meta)*0.8)][-1] # denotes end index
    
    train_X, train_y = full_X[:data_split], full_y[:data_split]
    test_X, test_y = full_X[data_split:], full_y[data_split:]
    
    rank_svm.train(train_X, train_y)
    rank_svm.save_model('rank_svm.p')
    
    rank_svm = RankSVM()
#    rank_svm.load_model('rank_svm.p')
    rank_svm = pickle.load(open('rank_svm.p','rb'))
    preds = rank_svm.test(test_X)
    train_lookup_snippets = pickle.load(open('pairwise_train_snippets_lookup.list','rb'))
#    print('Average Precision Score : %s ' %(average_precision_score(test_y, preds)))
#    pdb.set_trace()
    
    