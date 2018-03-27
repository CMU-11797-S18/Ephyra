import os

class MERTagger:
	def __init__(self):
		self.testfile = 'testfile.iob'

	def tag(self,sentences):
		iobString = ""
		for sent in sentences:
			for w in sent.split(" "):
				iobString+= str(w+" O\n")
			iobString+="\n"
		f = open('testfile.iob','w')
		f.write(iobString)
		f.close()
		command = "python infer.py --train ../dataset/NLPBA/train/train.eng --dev ../dataset/NLPBA/train/dev.eng --pre_emb ../embeddings/bio_nlp_vec/PubMed-shuffle-win-30.bin -W 675 -H 12 -D 0.5 --lower 1 -A 0 --tag_scheme iob -P 1 -S 1 -w 200 -K 2,3,4 -k 40,40,40"
		command += " --test "+ self.testfile
		os.system(command)
		latest_file = max(self.all_files_under('../evaluation/temp_result/'), key=os.path.getmtime)
		f = open(latest_file,'r')
		tagged_sentences,sent=[],[]
		for line in f.readlines():
			print(line)
			if line=="\n":
				tagged_sentences.append(sent)
				sent=[]
			else:
				word,org,pred = line.split(" ")
				if(pred!='O'):
					sent.append((word,pred))
		print(tagged_sentences)

	def all_files_under(self,path):
		"""Iterates through all files that are under the given path."""
		for cur_path, dirnames, filenames in os.walk(path):
			for filename in filenames:
				yield os.path.join(cur_path, filename)


tagger = MERTagger()
sentences = ["Is Hirschsprung disease a mendelian or a multifactorial disorder","Hirschsprung disease (HSCR) is a multifactorial, non-mendelian disorder in which rare high-penetrance coding sequence mutations in the receptor tyrosine kinase RET contribute to risk in combination with mutations at other genes"]
tagger.tag(sentences)
