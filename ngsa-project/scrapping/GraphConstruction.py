# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 15:46:49 2018

@author: SAMUEL
"""

import os

import os
from tqdm import tqdm
import json
import numpy as np
from scipy.sparse import dok_matrix, csr_matrix
import networkx as nx
import matplotlib.pyplot as plt



class GraphConstruction:
    
    def __init__(self):
        self.articles_link=np.array([])
        self.article_to_num={}
        self.citation_to_num={}
        self.n_article=0
        self.n_citation=0
        self.graph=None
        self.cid_article={}
        
    def construct_citation_graph(self,direct=True):
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

                        cit=[]
                        with open(os.path.join(articlesPath, articleFileName)) as articleJsonData:
                            articleData = json.load(articleJsonData)
                            cid=articleData['cid']
                            self.cid_article[cid]='-'.join(date)+'/articles/'+articleFileName
                            if cid not in self.article_to_num.keys():
                                self.article_to_num[cid]=self.n_article
                                self.article_to_num[self.n_article]=cid
                                self.n_article+=1
                            if direct==False:
                                if cid not in self.citation_to_num.keys():
                                    self.citation_to_num[cid]=self.n_citation
                                    self.n_citation+=1
                                cit.append(self.citation_to_num[cid])
                            if direct:
                                for l in articleData['links']:
                                    if l['text']!=' ':
                                        if type(l['cid'])==str:
                                            cid=l['cid']
                                            if cid in self.article_to_num.keys():
                                                cit.append(self.article_to_num[cid])
                            else:
                                for l in articleData['links']:
                                    if l['text']!=' ':
                                        if type(l['cid'])==str:
                                            cid=l['cid']
                                            if cid not in self.citation_to_num.keys():
                                                self.citation_to_num[cid]=self.n_citation
                                                self.n_citation+=1
                                            cit.append(self.citation_to_num[cid])
                                        else:
                                            if l['href'] not in self.citation_to_num.keys():
                                                self.citation_to_num[l['href']]=self.n_citation
                                                self.n_citation+=1
                                            cit.append(self.citation_to_num[l['href']])                                    
                        citations_pos.append(cit)
        if direct:
            res=dok_matrix((self.n_article,self.n_article))
            for i,liste in enumerate(citations_pos):
                for j in liste:
                    res[i,j]=1
                res[i,i]=0
            res=res+res.T     
            self.articles_link=res
        else:
            m=max(self.citation_to_num.values())+1
            res=dok_matrix((self.n_article,m))
            for i,liste in enumerate(citations_pos):
                for j in liste:
                    res[i,j]=1
            self.articles_link=csr_matrix(np.sign((res.dot(res.T)).toarray()))
        
        self.graph=nx.from_scipy_sparse_matrix(self.articles_link)
        
        
    def graph_description(self):
        n_nodes=self.graph.number_of_nodes()
        n_edges=self.graph.number_of_edges()
        n_cc=nx.number_connected_components(self.graph)
        print('number of nodes: '+str(n_nodes))
        print('number of edges: '+str(n_edges))
        print('number of connected components: '+str(n_cc))
        ccs=sorted(nx.connected_components(self.graph), key = len, reverse=True)
        print('largest and second largest CC size: '+str(len(ccs[0]))+' , '+str(len(ccs[1])))
        plt.figure()
        plt.yscale('log')
        plt.xscale('log')
        plt.scatter(range(len(ccs)),[len(i) for i in ccs],s=10)
        plt.title('Taille des connected components')
        d=np.array(sorted([d[1] for d in self.graph.degree()], reverse=True))
        p=np.unique(d,return_counts=True)
        plt.figure()
        plt.loglog(p[0],p[1])
        plt.title('degree distribution')
        
    def get_ccs(self):
        ccs=sorted(nx.connected_component_subgraphs(self.graph), key=len,reverse=True)
        res=[]
        for cc in ccs:
            res_int=[]
            for nodes in cc.nodes():
                res_int.append(self.article_to_num[nodes])
            res.append(res_int)
        return res
            


if __name__=='__main__':
    gc=GraphConstruction()  
    gc.construct_citation_graph()
    gc.graph_description()               
