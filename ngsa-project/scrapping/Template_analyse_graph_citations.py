# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 21:35:54 2018

@author: SAMUEL
"""

from GraphConstruction import GraphConstruction
from calculateTFIDF import TFIDFmanager


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