# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 15:46:49 2018

@author: SAMUEL
"""

import os
from tqdm import tqdm
import json
import numpy as np
from scipy.sparse import dok_matrix



class GraphConstruction:
    
    def __init__(self):
        self.articles_link=np.array([])
        self.article_to_num={}
        self.citation_to_num={}
        self.n_article=0
        self.n_citation=0
        
    def construct_citation_graph(self):
        citations_pos=[]
        rootPath = os.path.join(os.getcwd(), 'output/')

        for fileName in tqdm(os.listdir(rootPath)):
            folderPath = os.path.join(rootPath, fileName)
            
        
            if os.path.isdir(folderPath):
                date = fileName.split('-')
                if 'articles' in os.listdir(folderPath):
                    articlesPath = os.path.join(folderPath, 'articles/')
                
                    # articles
                    for articleFileName in os.listdir(articlesPath):
                        self.article_to_num[' '.join(date)+articleFileName]=self.n_article
                        self.n_article+=1
                        cit=[]
                        with open(os.path.join(articlesPath, articleFileName)) as articleJsonData:
                            articleData = json.load(articleJsonData)
                            
                            # 'article_2017-12-05_66' for ex
                            for l in articleData['links']:
                                if l['text']!=' ':
                                    if l['href'] not in self.citation_to_num.keys():
                                        self.citation_to_num[l['href']]=self.n_citation
                                        self.n_citation+=1
                                    cit.append(self.citation_to_num[l['href']])
                        citations_pos.append(cit)
        m=max(self.citation_to_num.values())+1
#        res=np.zeros((self.n_article,m))
        res=dok_matrix((self.n_article,m))
        for i,liste in enumerate(citations_pos):
            for j in liste:
                res[i,j]=1
        self.articles_link=res.dot(res.T)

                
    
