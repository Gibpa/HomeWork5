import numpy as np
def purity(Clusters,Y,K):

    N_points  = len(Clusters)

    Sum_Major_type = np.zeros((K,1))
    
    for i in range(K):
        Major_Type = np.zeros((5,1))
        for j in range(N_points):
            if (Clusters[j] == i):
                if ( Y[j] == 0):
                     Major_Type[0] += 1
                if ( Y[j] == 1):
                     Major_Type[1] += 1
                if ( Y[j] == 2):
                     Major_Type[2] += 1
                if ( Y[j] == 3):
                     Major_Type[3] += 1
                if ( Y[j] == 1):
                     Major_Type[4] += 1
        Sum_Major_type[i] = np.amax(Major_Type)
    Purity = np.sum(Sum_Major_type) / N_points

    return Purity