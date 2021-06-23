import numpy as np


def create_nl_feature(X):
    '''
    TODO - Create additional features and add it to the dataset
    
    returns:
        X_new - (N, d + num_new_features) array with 
                additional features added to X such that it
                can classify the points in the dataset.
    '''
    N, d = X.shape
    temp = []
    for i in range(N):
        temp_line = []
        # for feature in range(d):
        #     temp_line.append((-1) * X[i][feature])
        #     temp_line.append(X[i][feature] * X[i][feature])
        temp_line.append(X[i][0] * X[i][1])
        temp.append(temp_line)
    X_new = np.hstack((X, temp))
    return X_new