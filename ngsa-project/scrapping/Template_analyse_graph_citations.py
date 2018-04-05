# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 21:35:54 2018

@author: SAMUEL
"""

from GraphConstruction import GraphConstruction
from calculateTFIDF import TFIDFmanager
import networkx as nx
import community
import numpy as np
import json


gc=GraphConstruction()
### Construction du graph en fonction des citations directes
### Pour construire le graph en fonction des citations communes :
### gc.construct_citation_graph(direct=False)
### Déconseillé pour cause de temps de calcul pour l'instant
gc.construct_citation_graph()
gc.graph_description()   
 
### Liste des connected components du graph precedemment construit
Connected_components=gc.get_ccs()


analyser=TFIDFmanager()
### Analyse textuelle des différents articles
analyser.go_through_data()
for k in range(1,6):
    print('les 15 mots les plus représentatifs de la '+str(k+1)+'ème CC: ')
    dic_word_importance=analyser.tfidf_cc(Connected_components[k])
    print(TFIDFmanager.top_tfidf(dic_word_importance))
    
### Analyse de la connected component principale
L_cc=max(nx.connected_component_subgraphs(gc.graph), key=len)
partition = community.best_partition(L_cc)
n_sub_graph=len(np.unique(list(partition.values())))
nodes_cluster=np.array([[i,partition[i]] for i in partition.keys()])
sub_g_LCC={}
for i in range(n_sub_graph):
    list_nodes=nodes_cluster[:,0][nodes_cluster[:,1]==i]
#    print(list_nodes)
    list_cid=gc.conv_nodes_cid(list_nodes)
#    print(list_cid)
    print('les 15 mots les plus représentatifs de la '+str(i+1)+'ème cluster de taille '+str(len(list_cid))+': ')
    dic_word_importance=analyser.tfidf_cc(list_cid)
    print(TFIDFmanager.top_tfidf(dic_word_importance))
    
    
hdn=sorted(gc.graph.nodes(),key=gc.graph.degree)
print(gc.cid_article[gc.article_to_num[hdn[2]]])
json.load(open('output/'+gc.cid_article[gc.article_to_num[hdn[2]]]))


