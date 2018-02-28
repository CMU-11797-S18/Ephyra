import createData as data
#import operator
import pdb
import pickle

def evaluateResults(gold_ranks, pred_ranks):
    p1 = 0
    p3 = 0
    p5 = 0
    p10 = 0
    
    t1 = 0
    t3 = 0
    t5 = 0
    t10 = 0
    
    assert len(gold_ranks) == len(pred_ranks)
    
    for i in range(len(gold_ranks)):
#        pdb.set_trace()
        temp3 = 0
        temp5 = 0
        temp10 = 0
        # P@1
        if gold_ranks[i][0] == pred_ranks[i][0]:
            p1 += 1
        t1 += 1
            
        # P@3
        if len(gold_ranks[i]) >= 3:
            for j in range(3):
                if pred_ranks[i][j] in gold_ranks[i][:3]:
                    temp3 += 1
            p3 += temp3/3.
            t3 += 1
            
        # P@5
        if len(gold_ranks[i]) >= 5:
            for j in range(5):
                if pred_ranks[i][j] in gold_ranks[i][:5]:
                    temp5 += 1
            p5 += temp5/5.
            t5 += 1 
            
         # P@10
        if len(pred_ranks[i]) >= 10:
            for j in range(10):
                if pred_ranks[i][j] in gold_ranks[i][:10]:
                    temp10 += 1
            p10 += temp10/10.
            t10 += 1 
            
    return p1/float(t1), p3/t3, p5/t5, p10/t10
            
                

def createGT():
    cd = data.createData([])
    queries_all = pickle.load(open("data/queries.p","rb"))
    snippets_all = pickle.load(open("data/snippets.p","rb"))
    ideal_answers_all = pickle.load(open("data/ideal_answers.p","rb"))

    all_scores = []
    for i in range(len(queries_all)):
        query = queries_all[i]
        snippets = snippets_all[i]
        _ , BM25ScoresIdealAns = cd.getBM25(query, ideal_answers_all[i], snippets_all[i])
        score_list = [(j, BM25ScoresIdealAns[j]) for j in range(len(BM25ScoresIdealAns))]
        score_list.sort(key=lambda i: i[1]) #operator.itemgetter(1)
#        pdb.set_trace()
        score_gt = [score_list[j][0] for j in range(len(score_list))][::-1]
        if len(snippets) == 0:
            continue
        all_scores.append(score_gt)
    
    pickle.dump(all_scores,open("data/score_gt_unweighted.p","wb"))

if __name__ == '__main__':
    createGT()