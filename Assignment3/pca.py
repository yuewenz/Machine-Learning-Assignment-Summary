import numpy as np
import matplotlib.pyplot as plt

        
class PCA(object):

    def __init__(self):
        self.U = None
        self.S = None
        self.V = None

    def fit(self, X): # 5 points
        """
        Decompose dataset into principal components.
        You may use your SVD function from the previous part in your implementation or numpy.linalg.svd function.

        Don't return anything. You can directly set self.U, self.S and self.V declared in __init__ with
        corresponding values from PCA.

        Args:
            X: N*D array corresponding to a dataset
        Return:
            None
        """
        X = X - X.mean(axis=0)
        self.U, self.S, self.V = np.linalg.svd(X, full_matrices=False)

    def transform(self, data, K=2): # 2 pts
        """
        Transform data to reduce the number of features such that final data has given number of columns

        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: N*D array corresponding to a dataset
            K: Int value for number of columns to be kept
        Return:
            X_new: N*K array corresponding to data obtained by applying PCA on data
        """
        n = data.shape[0]
        self.fit(data)
        X_new = np.matmul(data, self.V[:K,:].transpose()) # X * V[:K,:]T
        return X_new

    def transform_rv(self, data, retained_variance=0.99): # 3 pts
        """
        Transform data to reduce the number of features such that a given variance is retained

        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: N*D array corresponding to a dataset
            retained_variance: Float value for amount of variance to be retained
        Return:
            X_new: N*K array corresponding to data obtained by applying PCA on data
        """
        self.fit(data)
        S = self.S ** 2
        prop = np.cumsum(S) / S.sum() # compute the proportion of variance
        K = np.where(prop >= retained_variance)[0][0]
        X_new = np.dot(data, self.V[:K+1,:].transpose())
        return X_new

    def get_V(self):
        """ Getter function for value of V """
        return self.V
    
    def visualize(self, X, y): 
        """
        Use your PCA implementation to reduce the dataset to only 2 features.

        Create a scatter plot of the reduced data set and differentiate points that
        have different true labels using color.

        Args:
            xtrain: NxD numpy array, where N is number of instances and D is the dimensionality 
            of each instance
            ytrain: numpy array (N,), the true labels
            
        Return: None
        """
        print("data shape before PCA ", X.shape)
        X = self.transform(X, K=2)
        print("data shape after PCA ", X.shape)
        colors = ['blue', 'red']
        y_u = np.unique(y)
        for i in range(len(y_u)):
            v = y_u[i]
            plt.scatter(X[y==v, 0], X[y==v, 1], color=colors[i], marker='x', label=v)
        
        ##################### END YOUR CODE ABOVE, DO NOT CHANGE BELOW #######################
        plt.legend()
        plt.show()