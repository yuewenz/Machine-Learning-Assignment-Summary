from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import tqdm
# Load image
import imageio


# Set random seed so output is all same
np.random.seed(1)

def pairwise_dist(x, y):
    return np.sqrt(np.sum((x[:,None]-y[None,:])**2,-1))

class KMeans(object):
    
    def __init__(self): #No need to implement
        pass

    def pairwise_dist(self,x, y):
        
        return np.sqrt(np.sum((x[:,None]-y[None,:])**2,-1))

    def _init_centers(self, points, K, **kwargs): # [5 pts]
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            kwargs: any additional arguments you want
        Return:
            centers: K x D numpy array, the centers. 
        """
        
        center_point=[]
        new_point=np.random.permutation(points.copy())
        for p in range(K):
            center_point.append(new_point[p])
        centers=np.array(center_point)
        return centers

        
        
        
        

        #raise NotImplementedError

    def _update_assignment(self, centers, points): # [10 pts]
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            points: NxD numpy array, the observations
        Return:
            cluster_idx: numpy array of length N, the cluster assignment for each point
            
        Hint: You could call pairwise_dist() function.
        """
        pdis=pairwise_dist(centers,points)
        cluster_idx=np.argmin(pdis,axis=0)
        return cluster_idx
        
        
    def _update_centers(self, old_centers, cluster_idx, points): # [10 pts]
        """
        Args:
            old_centers: old centers KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            centers: new centers, K x D numpy array, where K is the number of clusters, and D is the dimension.
        """
        K = len(old_centers)
        D = len(old_centers[0])
        
        #new_center_point=np.zeros((K,D))
        new_center_point=[[0]*D]*K
        L=[]
        #print(new_center_point)
        for i in range(K):
            cluster=np.array(points[cluster_idx==i])
            cluster_mean=np.mean(cluster,axis=0)
            new_center_point[i]=cluster_mean
            L.append(new_center_point[i])
        centers=np.array(L)
        return centers
            
                       
        #raise NotImplementedError
        
        

    def _get_loss(self, centers, cluster_idx, points): # [5 pts]
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            loss: a single float number, which is the objective function of KMeans. 
        """
      
        loss=0
        for i in range(len(centers)):
            x=points[cluster_idx==i]
            y=centers[i]
            pdis=((pairwise_dist(x,y))**2).sum()
            #print(pdis,i)
            loss=loss+pdis
        return loss
       
        
        
    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, verbose=False, **kwargs):
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            verbose: boolean to set whether method should print loss (Hint: helpful for debugging)
            kwargs: any additional arguments you want
        Return:
            cluster assignments: Nx1 int numpy array
            cluster centers: K x D numpy array, the centers
            loss: final loss value of the objective function of KMeans
        """
        centers = self._init_centers(points, K, **kwargs)
        for it in range(max_iters):
            cluster_idx = self._update_assignment(centers, points)
            centers = self._update_centers(centers, cluster_idx, points)
            loss = self._get_loss(centers, cluster_idx, points)
            K = centers.shape[0]
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            if verbose:
                print('iter %d, loss: %.4f' % (it, loss))
        return cluster_idx, centers, loss
    
def find_optimal_num_clusters(data, max_K=19): # [10 pts]
    np.random.seed(1)
    loss_values=[]
    K_number=[]
    for i in range(1,max_K+1):
        cluster_idx, centers, loss=KMeans().__call__(data, i)
        loss_values.append(loss)
        K_number.append(i)
    plt.plot(K_number,loss_values)
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()
    return loss_values
    """Plots loss values for different number of clusters in K-Means

    Args:
        image: input image of shape(H, W, 3)
        max_K: number of clusters
    Return:
        List with loss values
    """

    #raise NotImplementedError


def intra_cluster_dist(cluster_idx, data, labels): # [4 pts]
    """
    Calculates the average distance from a point to other points within the same cluster
    
    Args:
        cluster_idx: the cluster index (label) for which we want to find the intra cluster distance
        data: NxD numpy array, where N is # points and D is the dimensionality
        labels: 1D array of length N where each number indicates of cluster assignement for that point
    Return:
        intra_dist_cluster: 1D array where the i_th entry denotes the average distance from point i 

                   in cluster denoted by cluster_idx to other points within the same cluster
    
    """
    cluster=data[labels==cluster_idx,:]
    D=len(cluster)
    intra_dist_cluster = np.zeros(D)
    for x in range(len(cluster)):
        Y=[]
        for y in range(len(cluster)):
            if x!=y:
                #print(cluster[y])
                #print(x,y)
                Y.append(True)
            else:
                Y.append(False)
        pdis=pairwise_dist(cluster[Y,:],cluster[x]).mean()
        intra_dist_cluster[x]=pdis
    return intra_dist_cluster
                  
    
   

def inter_cluster_dist(cluster_idx, data, labels): # [4 pts]
    """
    Calculates the average distance from one cluster to the nearest cluster
    Args:
        cluster_idx: the cluster index (label) for which we want to find the intra cluster distance
        data: NxD numpy array, where N is # points and D is the dimensionality
        labels: 1D array of length N where each number indicates of cluster assignement for that point
    Return:
        inter_dist_cluster: 1D array where the i-th entry denotes the average distance from point i in cluster
                            denoted by cluster_idx to the nearest neighboring cluster
    """
    cal_cluster=data[labels==cluster_idx,:]
    uniqueValues = np.unique(labels)
    uniqueValues=np.delete(uniqueValues,cluster_idx)
    L=[]
    for i in uniqueValues:
        cluster=data[labels==i]
        intra_dist_cluster = np.zeros(len(cal_cluster))
        for a in range(len(cal_cluster)):
            pdis=pairwise_dist(cluster,cal_cluster[a]).mean()
            intra_dist_cluster[a]=pdis
        L.append(intra_dist_cluster)
    #print(L)
    return np.minimum.reduce(L)
    

def normalized_cut(data, labels): #[2 pts]
    """
    Finds the normalized_cut of the current cluster assignment
    
    Args:
        data: NxD numpy array, where N is # points and D is the dimensionality
        labels: 1D array of length N where each number indicates of cluster assignement for that point
    Return:
        normalized_cut: normalized cut of the current cluster assignment
    """
    uniqueValues = np.unique(labels)
    #a=0
    k=[]
    for i in uniqueValues:
        x=intra_cluster_dist(i, data, labels)
        y=inter_cluster_dist(i, data, labels)
        L=y/(x+y)
        k.append(L)
    nc=np.sum(np.sum(a) for a in k)
    #nc=np.add.reduce(k)
    return nc
    
    #raise NotImplementedError

