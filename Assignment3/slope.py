import numpy as np
from pca import PCA
from regression import Regression

class Slope(object):

    def __init__(self):
        pass

    def pca_slope(self, X, y):
        """
        Calculates the slope of the first principal component given by PCA

        Args: 
            x: (N,) vector of feature x
            y: (N,) vector of feature y
        Return:
            slope: Scalar slope of the first principal component
        """
        data = np.stack([X,y], axis=1)
        pca1 = PCA()
        pca1.transform(data)
        compo = pca1.get_V()[0]
        slope = compo[1] / compo[0]
        return slope
        
   
    def lr_slope(self, X, y):
        """
        Calculates the slope of the best fit as given by Linear Regression
        
        For this function don't use any regularization

        Args: 
            X: N*1 array corresponding to a dataset
            y: N*1 array of labels y
        Return:
            slope: slope of the best fit
        """
        weight = Regression().linear_fit_closed(X, y)
        return weight[0]

    def addNoise(self, c, x_noise = False, seed = 1):
        """
        Creates a dataset with noise and calculates the slope of the dataset
        using the pca_slope and lr_slope functions implemented in this class.

        Args: 
            c: Scalar, a given noise level to be used on Y and/or X
            x_noise: Boolean. When set to False, X should not have noise added
                     When set to True, X should have noise
            seed: Random seed
        Return:
            pca_slope_value: slope value of dataset created using pca_slope
            lr_slope_value: slope value of dataset created using lr_slope

        """
        np.random.seed(seed) #### DO NOT CHANGE THIS ####
        ############# START YOUR CODE BELOW #############
        x = np.linspace(0.001, 1, 1000)
        x_old=x
        if x_noise:
            noise = np.random.normal(loc=[0], scale=c, size=x.shape)
            x = x + noise
        noise = np.random.normal(loc=[0], scale=c, size=x.shape)
        #y = 2*x + noise
        y=2*x_old+noise
        pca_slope_value = self.pca_slope(x, y)
        lr_slope_value = self.lr_slope(x[:, None], y)
        return pca_slope_value, lr_slope_value












