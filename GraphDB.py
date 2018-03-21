from py2neo import authenticate,Graph, Path
from py2neo import Node, Relationship
from GraphConstants import Constants
import pdb

class GraphDatabase:
	def __init__(self, hostname="http://localhost:7474/", username="neo4j", password="Kalwad"):
		#TODO: Change Username and Password
		self.graph = Graph("http://localhost:7474",  username="neo4j", password="Kalwad")
		self.constants = Constants()

	def insertTriplet(self,verb=None,subjectEntity=None,objectEntity=None,subject_adj=[],object_adj=[],pubmedId=None,offset=None):
		subject_node = self.queryNode(subjectEntity)
		if(subject_node==None):
			subject_node = Node(self.constants.nounEntity,name=subjectEntity)

		object_node = self.queryNode(objectEntity)
		if(object_node==None):
			object_node = Node(self.constants.nounEntity,name=objectEntity)

		for a in subject_adj:
			adj_node = Node(self.constants.adjectiveEntity,name=a)
			assoc = Relationship(adj_node, self.constants.adjectiveRelationship, subject_node)
			self.graph.create(assoc)

		for a in object_adj:
			adj_node = Node(self.constants.adjectiveEntity,name=a)
			assoc = Relationship(adj_node, self.constants.adjectiveRelationship, object_node)
			self.graph.create(assoc)

		# pdb.set_trace()
		newassoc = self.queryRelationship(subject_node=subjectEntity,object_node=objectEntity,verb=verb)
		if(newassoc==None):
			newassoc = Relationship(subject_node, verb, object_node,location=str(pubmedId)+"|"+str(offset))
			self.graph.merge(newassoc)
		else:
			for rel in self.graph.match(start_node=subject_node, rel_type=verb, end_node=object_node):
				print(rel.properties['location'])
				newlocations = set()
				locations = rel.properties['location'].split(",")
				for loc in locations:
					newlocations.add(loc)
				newlocations.add(str(str(pubmedId)+"|"+str(offset)))
				prop = ",".join(newlocations)
				rel.properties['location'] = prop
				print(rel.properties['location'])
				self.graph.create(rel)

	def fullQuery(self,verb=None,named_entities=None,medical_entities=None):
		q = 'MATCH (u:{})-[r:{}]->(m:{}) WHERE u.name="{}" RETURN u,r,m'.format(self.constants.nounEntity,verb,self.constants.nounEntity,named_entities)
		results = self.graph.run(q)
		for r in results:
		    print("(%s)-[%s]->(%s)" % (r[0]["name"], r[1], r[2]["name"]))
		return results

	def queryNode(self,node=None):
		q = 'MATCH (u:{}) WHERE u.name="{}" RETURN u'.format(self.constants.nounEntity,node)
		resultNode = self.graph.evaluate(q)
		print(type(resultNode))
		return resultNode

	def queryRelationship(self,subject_node=None,verb=None,object_node=None):
		q = 'MATCH (u:{})-[r:{}]->(m:{}) WHERE u.name="{}" AND m.name="{}" RETURN r'.format(
			self.constants.nounEntity,verb,self.constants.nounEntity,subject_node,object_node)
		resultRel = self.graph.evaluate(q)
		print((resultRel))
		# resultRel = self.graph.match(start_node=subject_node, rel_type="CHECKED_OUT", end_node=object_node)
		# print(resultRel)
		return resultRel

	def getIncomingRelationships(self,objectEntity=None,verb=None):
		object_node = self.queryNode(objectEntity)
		relations = []
		for rel in self.graph.match(rel_type=verb, end_node=object_node):
			realtions.add(rel)
		return relations


#TESTING 
graph = GraphDatabase()
graph.insertTriplet(verb="cures",subjectEntity="Turmeric",objectEntity="infections",pubmedId=1234,offset=10)
# graph.query(verb="cures",named_entities="Neem")
# graph.queryRelationship(node="Neem",verb="cures")
# graph.getIncomingRelationships(verb="cures",objectEntity="infections")
