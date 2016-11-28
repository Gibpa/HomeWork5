#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:40:41 2016

@author: markno1
"""

from sklearn import datasets
from plot import plot_2D_decision_regions_GMM,plot_2D_decision_regions_Kmean
digits = datasets.load_digits()

#plt.imshow(digits.images[5])
#plt.show()




X = digits.data
y = digits.target

X = X[y<5]
y = y[y<5]

from sklearn import preprocessing
X = preprocessing.scale(X)


from sklearn.decomposition import PCA
clf = PCA(n_components=2)
X_t = clf.fit_transform(X)



from sklearn.cluster import KMeans

for i in range(3,11):

    kmeans = KMeans(i)
    kmeans.fit(X_t)
    
    plot_2D_decision_regions_Kmean(X_t,kmeans,0.2,"K-mean k= "+str(i))
    
    from sklearn.metrics import normalized_mutual_info_score, homogeneity_score
    norm_mutual = normalized_mutual_info_score(y,kmeans.predict(X_t))
    hom_geneity = homogeneity_score(y,kmeans.predict(X_t))
    print("Nomalized mutual Info = "+str(norm_mutual))
    print("Homogeneity score = "+str(hom_geneity))
    

from sklearn import mixture

for i in range(2,11):
    mixture.GaussianMixture
    
    # Fit a Gaussian mixture with EM using five components
    gmm = mixture.GaussianMixture(n_components=i, covariance_type='full').fit(X_t)
    plot_2D_decision_regions_GMM(X_t,gmm,0.2,"Gaussian Mixture component = "+str(i))
    
    norm_mutual = normalized_mutual_info_score(y,gmm.predict(X_t))
    hom_geneity = homogeneity_score(y,gmm.predict(X_t))
    print("Nomalized mutual Info = "+str(norm_mutual))
    print("Homogeneity score = "+str(hom_geneity))