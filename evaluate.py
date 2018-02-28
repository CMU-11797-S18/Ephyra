import createData as data
import operator
import pdb
import pickle

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
		score_list.sort(key=operator.itemgetter(1))
		score_gt = [score_list[j][0] for j in range(len(score_list))]
		if len(snippets) == 0:
			continue
		all_scores.append(score_gt)
	
	pickle.dump(all_scores,open("data/score_gt_unweighted.p","wb"))

createGT()