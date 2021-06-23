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


SIGMA_CONST = 1e-6
LOG_CONST = 1e-32

class GMM(object):
    def __init__(self, X, K, max_iters = 100): # No need to change
        """
        Args: 
            X: the observations/datapoints, N x D numpy array
            K: number of clusters/components
            max_iters: maximum number of iterations (used in EM implementation)
        """
        self.points = X
        self.max_iters = max_iters
        
        self.N = self.points.shape[0]        #number of observations
        self.D = self.points.shape[1]        #number of features
        self.K = K                           #number of components/clusters

    #Helper function for you to implement
    def softmax(self, logit): # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array. See the above function.
        """
        tmp = np.max(logit, axis=1)
        logit -= tmp.reshape((logit.shape[0], 1))
        logit_exp = np.exp(logit)
        prob = logit_exp / np.sum(logit_exp, axis=1).reshape((logit.shape[0],1))
        logit += tmp.reshape((logit.shape[0], 1))
        return prob

    def logsumexp(self, logit): # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:]). See the above function
        """
        tmp = np.max(logit, axis=1)
        logit -= tmp.reshape((logit.shape[0], 1))
        s = np.log(np.sum(np.exp(logit), axis=1)) + tmp
        logit += tmp.reshape((logit.shape[0], 1))
        s = np.expand_dims(s, -1)
        return s

    #for undergraduate student
    def normalPDF(self, logit, mu_i, sigma_i): #[5pts]
        """
        Args: 
            logit: N x D numpy array
            mu_i: 1xD numpy array, the center for the ith gaussian.
            sigma_i: 1xDxD numpy array, the covariance matrix of the ith gaussian.  
        Return:
            pdf: 1xN numpy array, the probability distribution of N data for the ith gaussian
            
        Hint: 
            np.diagonal() should be handy.
        """
        
        #raise NotImplementedError
    
    #for grad students
    def multinormalPDF(self, logits, mu_i, sigma_i):  #[5pts]
        """
        Args: 
            logit: N x D numpy array
            mu_i: 1xD numpy array, the center for the ith gaussian.
            sigma_i: 1xDxD numpy array, the covariance matrix of the ith gaussian.  
        Return:
            normal_pdf: 1xN numpy array, the probability distribution of N data for the ith gaussian
            
        Hint: 
            np.linalg.det() and np.linalg.inv() should be handy.
            Add SIGMA_CONST if you encounter LinAlgError
        """
        N, D = logits.shape
        sigma_i_sqz = sigma_i[0]
        tmp = np.matmul((logits - mu_i), np.linalg.inv(sigma_i+SIGMA_CONST)).T * (logits - mu_i).T # D×N
        tmp = np.sum(tmp, axis=0)
        normal_pdf = 1.0 / np.power(2.0 * np.pi, D / 2.0) * \
                        np.power(np.linalg.det(sigma_i), - 0.5) * \
                        np.exp(-0.5 * tmp)
        return normal_pdf

    def _init_components(self, **kwargs): # [5pts]
        """
        Args:
            kwargs: any other arguments you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. 
                You will have KxDxD numpy array for full covariance matrix case

        Hint: You can use KMeans to initialize the centers for each gaussian
        """
        pi = np.ones(self.K) * (1.0 / self.K)
        rand_idx = np.random.choice(self.N, size=self.K, replace=False)
        mu = self.points[rand_idx]
        sigma = np.array([np.eye(self.points.shape[1]) for _ in range(self.K)])
        return pi, mu, sigma

    def _ll_joint(self, pi, mu, sigma, full_matrix = True, **kwargs): # [10 pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
            
        Return:
            ll(log-likelihood): NxK array, where ll(i, k) = log pi(k) + log NormalPDF(points_i | mu[k], sigma[k])
            
        Hint: You might find it useful to add LOG_CONST to the expression before taking the log 
        """
        ll_ls = []
        for k in range(self.K):
            ll_k = np.log(pi[k]+LOG_CONST) + np.log(self.multinormalPDF(self.points, mu[k], sigma[k])+LOG_CONST)
            ll_ls.append(ll_k)
        ll = np.array(ll_ls).T
        return ll

    def _E_step(self, pi, mu, sigma, **kwargs): # [5pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
            array for full covariance matrix case
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            
        Hint: 
            You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above. 
        """
        gamma = self.softmax(self._ll_joint(pi, mu, sigma))
        return gamma

    def _M_step(self, gamma, full_matrix = True, **kwargs): # [10pts]
        """
        Args:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            
        Hint:  
            There are formulas in the slide and in the above description box.
        """
        N_k = np.sum(gamma, axis=0)
        mu = np.array(
            [np.sum(gamma[:, k].reshape(self.N, 1) * self.points, axis=0) / N_k[k] for k in range(self.K)] 
        )
        sigma = np.array(
            [np.matmul((gamma[:, k].reshape(self.N, 1) * (self.points - mu[k].reshape(1, self.D))).T, self.points - mu[k].reshape(1, self.D)) / N_k[k]
            for k in range(self.K)]
            )
        pi = N_k / self.N
        return pi, mu, sigma
    
    def __call__(self, full_matrix = True, abs_tol=1e-16, rel_tol=1e-16, **kwargs): # No need to change
        """
        Args:
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want
        
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)       
        
        Hint: 
            You do not need to change it. For each iteration, we process E and M steps, then update the paramters. 
        """
        pi, mu, sigma = self._init_components(**kwargs)
        pbar = tqdm(range(self.max_iters))
        
        for it in pbar:
            # E-step
            gamma = self._E_step(pi, mu, sigma)
            
            # M-step
            pi, mu, sigma = self._M_step(gamma, full_matrix)
            
            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(pi, mu, sigma, full_matrix)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description('iter %d, loss: %.4f' % (it, loss))
        return gamma, (pi, mu, sigma)
