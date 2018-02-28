from __future__ import print_function
from gensim.summarization.bm25 import get_bm25_weights
import numpy as np 
import scipy
import string
import itertools
from preprocess import *
import pdb
import pickle

query_w2v = np.load("data/queries_w2v.npy")
snippets_w2v = np.load("data/snippets_w2v.npy")
        
class createData():
    """docstring for createData"""
    def __init__(self, featureList):
        self.availableFeatures = ["BM25", "cosSim"]
        self.featureList = featureList
        self.QueryTFIDF, self.SnippetsTFIDF, self.SnippetsLookup = createTFIDF()


    def getTFIDF(self, docListLen, idx):
        queryRep = self.QueryTFIDF[idx,:]
        snippetRep = self.SnippetsTFIDF[self.SnippetsLookup[idx] : self.SnippetsLookup[idx] + docListLen, :]

        return queryRep, snippetRep

    def getBM25(self, query, ideal_ans, docList):
        scores = np.array(get_bm25_weights(query+ideal_ans+docList))
        BM25ScoresQuery = np.array(scores[0][2:])
        BM25ScoresIdealAns = np.array(scores[1][2:])
#        pdb.set_trace()
        return BM25ScoresQuery,BM25ScoresIdealAns

    def getCosSim(self, query, docList, idx):
        cosSimVec = []

        queryTFIDF, docTFIDF = self.getTFIDF(len(docList), idx)
        a = queryTFIDF.todense()
#        pdb.set_trace()
        for i in range(docTFIDF.shape[0]):
            b = docTFIDF.getrow(i).todense()
#            except:
#                pdb.set_trace()
            cosSimVec.append(scipy.spatial.distance.cosine(a,b))
#        pdb.set_trace()
        cosSimVec = np.vstack(cosSimVec)
        return cosSimVec
    
    def get_tfidf_dot(self, docList, idx):
        tfidfVec = []
#        pdb.set_trace()
        queryTFIDF, docTFIDF = self.getTFIDF(len(docList), idx)
        a = queryTFIDF
        for i in range(docTFIDF.shape[0]):
            b = docTFIDF.getrow(i)
            tfidfVec.append(a.dot(b.T)[0,0])

        tfidfVec = np.vstack(tfidfVec)
        return tfidfVec
    
    def getWord2Vec(self, query, docList, idx):
        w2v = []
        queryW2V = query_w2v[idx]
        snippetsW2V = snippets_w2v[idx: idx+len(docList), :]
        
        for i in range(len(docList)):
            snippetW2V = snippetsW2V[i]
            w2v.append(scipy.spatial.distance.cosine(queryW2V,snippetW2V))
        
        w2v = np.vstack(w2v)
        return w2v
        

    def getFeatureVectors(self, query, idealAns, docList, idx):
        featureVec = []
        bm25 = self.getBM25(query,idealAns, docList)
        for feature in self.featureList:
            if feature == "cosSim":
                cosSim = self.getCosSim(query,docList, idx)
                featureVec.append(cosSim)
            if feature == "tfidfDot":
                tfidf = self.get_tfidf_dot(docList, idx)
                featureVec.append(tfidf)

        featureVec = np.hstack(featureVec)
        if "BM25" in self.featureList:
            return bm25, featureVec
        else:
            return bm25, featureVec

def create_pairwise_dataset(queries, ideal_answers, snippets, featureList):
    '''
    Method which creates the pairwise dataset for RankSVM
    '''
    cd = createData(featureList)
    bm25_flag = False
    if "BM25" in featureList:
        bm25_flag = True
    
    X = []
    y = []
    train_snippets_lookup = []
    count = 0
    for i in range(len(queries)):
#        pdb.set_trace()
        num_snippets = len(snippets[i])
        if num_snippets > 0:
            bm25, featureVec = cd.getFeatureVectors(queries[i], ideal_answers[i], snippets[i], i)
#            pdb.set_trace()
            snippet_pairs = itertools.permutations(range(featureVec.shape[0]),2)
            train_snippets_lookup.append((i, featureVec.shape[0]))
    #        pdb.set_trace()
            for j,k in snippet_pairs:
                if bm25_flag is True:
    #                pdb.set_trace()
                    x1 = np.array([bm25[0][j]] + featureVec[j].tolist())
                    x2 = np.array([bm25[0][k]] + featureVec[k].tolist())
                    X.append(x1-x2)
                    
                    if 0.8*bm25[1][j]+0.2*bm25[0][j] > bm25[1][k]+0.2*bm25[0][k]:
                        y.append(1.)
                    else:
                        y.append(-1.)
                    
                else:
    #                pdb.set_trace()
                    x1 = np.array([bm25[j]] + featureVec[j].tolist())
                    x2 = np.array([bm25[k]] + featureVec[k].tolist())
                    X.append(x1-x2)
                    
                    if 0.8*bm25[1][j]+0.2*bm25[0][j] > bm25[1][k]+0.2*bm25[0][k]:
                        y.append(1.)
                    else:
                        y.append(-1.)
                    
        count += 1
        print(count)
    
    X = np.array(X)
    y = np.array(y)
    np.save('pairwise_trainX_6b.npy', X)
    np.save('pairwise_trainy_6b.npy', y)
    pickle.dump(train_snippets_lookup, open('pairwise_train_snippets_lookup.list','wb'))
    print('X.shape : ', X.shape)
    print('y.shape : ', y.shape)
    

if __name__ == '__main__':
    
    queries_all = pickle.load(open("data/queries.p","rb"))
    snippets_all = pickle.load(open("data/snippets.p","rb"))
    ideal_answers_all = pickle.load(open("data/ideal_answers.p","rb"))
    featureList = ["BM25","cosSim","tfidfDot"]
#    bm25_all = []
#    cosSim_all = []
#    cd = createData(featureList)
    create_pairwise_dataset(queries_all, ideal_answers_all, snippets_all, featureList)
#    for i in range(len(queries_all)):
#        query = queries_all[i]
#        snippets = snippets_all[i]
#        cosSim_all.append(cd.getFeatureVectors(query, snippets, i))

#    pickle.dump(cosSim_all,open("data/BM25_all.p","wb"))
    
#    bm25_all = pickle.load(open("data/BM25_all.p","rb"))
    train_snippets_lookup = pickle.load(open('pairwise_train_snippets_lookup.list'))
    train_snippets_meta = [(0,0,0,0)]
    for query_idx, num_snippets in train_snippets_lookup:
        #pdb.set_trace()
        data_idx_s = train_snippets_meta[-1][-1]
        data_idx_e = data_idx_s + (num_snippets * (num_snippets-1)) 
        train_snippets_meta.append((query_idx, num_snippets, data_idx_s, data_idx_e))

    del train_snippets_meta[0]

    pickle.dump(train_snippets_meta, open('train_snippets_meta.list','wb'))
    
