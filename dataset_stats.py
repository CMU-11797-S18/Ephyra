import json
import pdb
from StringIO import StringIO
data = json.loads(open("BioASQ-trainingDataset6b.json","r").read())

quesns = data['questions']
yes_no = []
summary = []
list = []
factoid = []
for q in quesns:
	if q[u'type'] == 'summary':
		try:
			summary.append(len(q['snippets'])*1.0)
		except:
			summary.append(0)
	elif q[u'type'] == 'list':
		try:
			list.append(len(q['snippets'])*1.0)
		except:
			list.append(0)
	elif q[u'type'] == 'factoid':
		try:
			factoid.append(len(q['snippets'])*1.0)
		except:
			factoid.append(0)
	else:
		try:
			yes_no.append(len(q['snippets'])*1.0)
		except:
			yes_no.append(0)

pdb.set_trace()
print("-"*100)
print("Summary")
print("-"*100)
print("Total:",len(summary),"Max number of snippets:",max(summary),"Min number of snippets:",min(summary),"Average number:",sum(summary)/len(summary))
print("-"*100)
print("Yes/No")
print("-"*100)
print("Total:",len(yes_no),"Max number of snippets:",max(yes_no),"Min number of snippets:",min(yes_no),"Average number:",sum(yes_no)/len(yes_no))
print("-"*100)
print("Factoid")
print("-"*100)
print("Total:",len(factoid),"Max number of snippets:",max(factoid),"Min number of snippets:",min(factoid),"Average number:",sum(factoid)/len(factoid))
print("-"*100)
print("List")
print("-"*100)
print("Total:",len(list),"Max number of snippets:",max(list),"Min number of snippets:",min(list),"Average number:",sum(list)/len(list))
