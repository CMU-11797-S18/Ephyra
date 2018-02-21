from gensim.summarization.bm25 import get_bm25_weights
import numpy as np 
import scipy
import string
from preprocess import *

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

	def getBM25(self, query, docList):
		scores = np.array(get_bm25_weights(query+docList))
		BM25Scores = np.array(scores[0][1:])
		
		return BM25Scores

	def getCosSim(self, query, docList, idx):
		cosSimVec = []

		queryTFIDF, docTFIDF = self.getTFIDF(len(docList), idx)
		for i in range(len(docList)):
			a = queryTFIDF.todense()
			b = docTFIDF.getrow(i).todense()
			cosSimVec.append(scipy.spatial.distance.cosine(a,b))

		cosSimVec = np.vstack(cosSimVec)
		return cosSimVec

	def getFeatureVectors(self, query, docList, idx):
		featureVec = []
		for feature in featureList:
			if feature =="BM25":
				bm25 = self.getBM25(query,docList)
				featureVec.append(bm25)
			if feature == "cosSim":
				cosSim = self.getCosSim(query,docList, idx)
				featureVec.append(cosSim)

		featureVec = np.hstack(featureVec)

		return featureVec


if __name__ == '__main__':
	
	queries_all = pickle.load(open("queries.p","rb"))
	snippets_all = pickle.load(open("snippets.p","rb"))

	featureList = ["cosSim"]
	#bm25_all = []
	cosSim_all = []
	cd = createData(featureList)
	for i in range(len(queries_all)):
		query = queries_all[i]
		snippets = snippets_all[i]
		cosSim_all.append(cd.getFeatureVectors(query, snippets, i))

	pickle.dump(cosSim_all,open("cosSim_all.p","wb"))
