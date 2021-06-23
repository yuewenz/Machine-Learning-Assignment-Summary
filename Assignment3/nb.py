'''
File: nb.py
Project: Downloads
File Created: March2021
Author: Yuzi Hu (yhu495@gatech.edu)
'''

import numpy as np
from sklearn.feature_extraction import text
from sklearn.model_selection import train_test_split
import pandas as pd

RANDOM_SEED = 5


class NaiveBayes(object):
    def __init__(self):
        pass

    def likelihood_ratio(self, X_negative, X_neutral, X_positive):  # [5pts]
        '''
        Args:
            X_negative: N_negative x D where N_negative is the number of negative news that we have,
                while D is the number of features (we use the word count as the feature)
            X_neutral: N_neutral x D where N_neutral is the number of neutral news that we have,
                while D is the number of features (we use the word count as the feature)
            X_positive: N_positive x D where N_positive is the number of positive news that we have,
                while D is the number of features (we use the word count as the feature)
        Return:
            likelihood_ratio: 3 x D matrix of the likelihood ratio of different words for different class of news
        '''
        lst = [X_negative, X_neutral, X_positive]
        llr = [(X.sum(axis=0) + 1) / (X.sum() + X.shape[1]) for X in lst]
        llr = np.stack(llr)
        return llr
        

    def priors_prob(self, X_negative, X_neutral, X_positive):  # [5pts]
        '''
        Args:
            X_negative: N_negative x D where N_negative is the number of negative news that we have,
                while D is the number of features (we use the word count as the feature)
            X_neutral: N_neutral x D where N_neutral is the number of neutral news that we have,
                while D is the number of features (we use the word count as the feature)
            X_positive: N_positive x D where N_positive is the number of positive news that we have,
                while D is the number of features (we use the word count as the feature)
        Return:
            priors_prob: 1 x 3 matrix where each entry denotes the prior probability for each class
        '''
        prior = np.array([X_negative.sum(), X_neutral.sum(), X_positive.sum()]).reshape((1,-1))
        return prior / prior.sum()

    # [5pts]
    def analyze_sentiment(self, likelihood_ratio, priors_prob, X_test):
        '''
        Args:
            likelihood_ratio: 3 x D matrix of the likelihood ratio of different words for different class of news
            priors_prob: 1 x 3 matrix where each entry denotes the prior probability for each class
            X_test: N_test x D bag of words representation of the N_test number of news that we need to analyze its sentiment
        Return:
             1 x N_test list, each entry is a class label indicating the sentiment of the news (negative: 0, neutral: 1, positive: 2)
        '''
        post = []
        for i in range(len(X_test)):
            llr = np.power(likelihood_ratio,X_test[i])
            #llr[llr==0] = 1
            llr = np.prod(llr, axis=1)
            post_probs = (llr * priors_prob).argmax()#[:,0]
            post.append(post_probs)
        #post_probs = np.stack(post)
        return post
