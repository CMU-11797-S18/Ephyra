from __future__ import print_function
import json
import re
import pdb
import string
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
from scipy.sparse import save_npz, load_npz


dataFile = json.loads(open("data/BioASQ-trainingDataset6b.json","r").read().strip())
#word_vectors = KeyedVectors.load_word2vec_format('models/wikipedia-pubmed-and-PMC-w2v.bin', binary=True)

def getAveragedWordVectors(doc_list):
    '''
    Creates an averaged word vector representation of the list of documents
    Args: list of words
    Returns: the averaged word vector representation of the doc_list as a matrix
    '''
    avg_arr = np.zeros((len(doc_list), word_vectors.vector_size))
    for i in range(len(doc_list)):
         for word in doc_list[i]:
             tot_w2v_embedding = np.zeros(word_vectors.vector_size)
             c = 0
             if word in word_vectors.vocab:
                 tot_w2v_embedding += word_vectors[word]
                 c += 1
         avg_arr[i] = tot_w2v_embedding/c
    
    return avg_arr
    
def createDump():
    '''
    Creates a dump of queries, snippets and documents in 
    '''
    queries = []
    ideal_answers = []
    snippets = []
    documents = []
    queries_w2v = []
    snippets_w2v = []
#    documents_w2v = []
    count = 0
    questions = dataFile['questions']
    for query in questions:
        queryVal = re.sub("["+string.punctuation.replace("\'","\"")+"]","",query['body']).lower().split()
        queries.append([queryVal])
#        pdb.set_trace()
        idealAnsVal = re.sub("["+string.punctuation.replace("\'","\"")+"]","",query['ideal_answer'][0]).lower().split()
        ideal_answers.append([idealAnsVal])
        
        queries_w2v.append(getAveragedWordVectors([queryVal]))
        if 'snippets' in query.keys():
            snippetsVal = [re.sub("["+string.punctuation.replace("\'","\"")+"]","",i['text']).lower().split() for i in query['snippets']]
            documentsVal = [re.sub("["+string.punctuation.replace("\'","\"")+"]","",i['document']).lower().split() for i in query['snippets']]
            snippets.append(snippetsVal)
            documents.append(documentsVal)
            snippets_w2v.append(getAveragedWordVectors(snippetsVal))
#            pdb.set_trace()
#            documents_w2v.append(documentsVal)
        else:
            snippets.append([])
            documents.append([])
            snippets_w2v.append(np.zeros((50,word_vectors.vector_size)))
#            documents_w2v.append(np.zeros((50,word_vectors.vector_size)))
        
        count += 1
        print(count)

    pickle.dump(queries,open("data/queries.p","wb"))
    pickle.dump(ideal_answers,open("data/ideal_answers.p","wb"))
    pickle.dump(snippets,open("data/snippets.p","wb"))
    pickle.dump(documents,open("data/documents.p","wb"))
    np.save(open("data/queries_w2v.npy","w"),queries_w2v)
    np.save(open("data/snippets_w2v.npy","w"),snippets_w2v)
#    np.save(open("data/documents_w2v.npy","w"),documents_w2v)

def createTFIDF():
    '''
    Creates TFIDF vectors
    '''
    '''
    queries = []
    snippets = []
    snippets_lookup = [0]
    questions = dataFile['questions']
    for query in questions:
        queries.append(query['body'])
        if 'snippets' in query.keys():
            snippets.extend([i['text'] for i in query['snippets']])
            if len(snippets_lookup) == 0:
                continue
            else:
#                pdb.set_trace()
                snippets_lookup.append(len(query['snippets'])+snippets_lookup[-1])
        else:
            snippets_lookup.append(snippets_lookup[-1]+1)
            snippets.extend([])

    total_data = queries+snippets
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')

    X_train = vectorizer.fit_transform(total_data)
    queries_tfidf = X_train[:len(queries),:]
    snippets_tfidf = X_train[len(queries):,:]

#      del snippets_lookup[0]
    np.save(open("data/snippets_lookup.npy", "wb"), snippets_lookup)
    
    save_npz(open("data/queries_tfidf.npz","wb"),queries_tfidf)
    save_npz(open("data/snippets_tfidf.npz","wb"),snippets_tfidf)
    '''
    queries_tfidf = load_npz("data/queries_tfidf.npz")
    snippets_tfidf = load_npz("data/snippets_tfidf.npz")
    snippets_lookup = np.load("data/snippets_lookup.npy")
    
    return queries_tfidf, snippets_tfidf, snippets_lookup

if __name__ == '__main__':
#    createDump()
    createTFIDF()