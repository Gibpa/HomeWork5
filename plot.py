#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 18:32:07 2016

@author: markno1
"""
import numpy as np
import matplotlib.pyplot as plt

colors = plt.cm.cool

#Input 
#      X      -  Data
#      clf    -  classificator
#      res    -  Resolution of the boundary
#      title  -  Title

def plot_2D_decision_regions_Kmean(X,kmeans,res,title):
    plt.figure(figsize=(9,5))
    plt.title(title)
    plt.grid(True)
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, res),
                         np.arange(y_min, y_max, res))

    Z = kmeans.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.2, color=colors)
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.predict(X), cmap=colors)
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],marker='x', s=169, linewidths=3,color='r', zorder=10)
    
    plt.savefig("/Users/marcotreglia/.bin/ML/Homework1/graphic/"+title+".png",dpi=100 )
    plt.legend(loc='upper left',shadow=True)
    plt.show()
    plt.close()
    
#Input 
#      X      -  Data
#      clf    -  classificator
#      res    -  Resolution of the boundary
#      title  -  Title

def plot_2D_decision_regions_GMM(X,gmm,res,title):
    plt.figure(figsize=(9,5))
    plt.title(title)
    plt.grid(True)
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, res),
                         np.arange(y_min, y_max, res))

    Z = gmm.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.2, color=colors)
    plt.scatter(X[:, 0], X[:, 1], c=gmm.predict(X), cmap=colors)
    
    centroids = gmm.means_
    plt.scatter(centroids[:, 0], centroids[:, 1],marker='x', s=169, linewidths=3,color='r', zorder=10)
    
    plt.savefig("/Users/marcotreglia/.bin/ML/Homework1/graphic/"+title+".png",dpi=100 )
    plt.legend(loc='upper left',shadow=True)
    plt.show()
    plt.close()
    
    
    #Input 
#      X      -  Data
#      label    -  label for the matching with the class
#      title  -  Title

    
def plot_2D(X,label,title):
    plt.figure(figsize=(9,5))
    plt.grid(True)
    plt.title(title)
    plt.scatter(X[:, 0], X[:, 1], c=label, cmap=colors)
    plt.savefig("/Users/marcotreglia/.bin/ML/Homework1/graphic/"+title+".png",dpi =100)
    plt.legend(loc='upper left',shadow=True)
    plt.show()
    plt.close()
