import numpy as np


class Regression(object):

    def __init__(self):
        pass

    def rmse(self, pred, label):  # [5pts]
        '''
        This is the root mean square error.
        Args:
            pred: numpy array of length N * 1, the prediction of labels
            label: numpy array of length N * 1, the ground truth of labels
        Return:
            a float value
        '''
        return np.sqrt(np.square(pred - label).mean())

    def construct_polynomial_feats(self, x, degree):  # [5pts]
        """
        Args:
            x: numpy array of length N, the 1-D observations
            degree: the max polynomial degree
        Return:
            feat: numpy array of shape Nx(degree+1), remember to include
            the bias term. feat is in the format of:
            [[1.0, x1, x1^2, x1^3, ....,],
             [1.0, x2, x2^2, x2^3, ....,],
             ......
            ]
        """
        if len(x.shape)==1:
            feats = np.zeros((len(x), degree+1)) + 1
            for i in range(degree):
                feats[:, i+1] = x ** (i+1)
        else:
            feats = np.zeros((x.shape[0], degree+1, x.shape[1]))
            for i in range(x.shape[0]):
                feats[i,0,:] = 1
                for j in range(degree):
                    feats[i, j+1, :] = x[i, :] ** (j+1)
        return feats

    def predict(self, xtest, weight):  # [5pts]
        """
        Args:
            xtest: NxD numpy array, where N is number
                   of instances and D is the dimensionality of each
                   instance
            weight: Dx1 numpy array, the weights of linear regression model
        Return:
            prediction: Nx1 numpy array, the predicted labels
        """
        return np.dot(xtest, weight)

    # =================
    # LINEAR REGRESSION
    # Hints: in the fit function, use close form solution of the linear regression to get weights.
    # For inverse, you can use numpy linear algebra function
    # For the predict, you need to use linear combination of data points and their weights (y = theta0*1+theta1*X1+...)

    def linear_fit_closed(self, xtrain, ytrain):  # [5pts]
        """
        Args:
            xtrain: N x D numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: N x 1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        tmp = np.linalg.pinv(np.dot(xtrain.T, xtrain))
        sol = np.dot(np.dot(tmp, xtrain.T), ytrain)
        return sol

    def linear_fit_GD(self, xtrain, ytrain, epochs=5, learning_rate=0.001):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        # weights initialization
        weight = np.zeros((xtrain.shape[1],1))
        for _ in range(epochs):
            # calculate gradient
            grad = 2/xtrain.shape[0] * xtrain.T.dot(xtrain.dot(weight) - ytrain)
            grad *= learning_rate
            weight -= grad
        return weight         

    def linear_fit_SGD(self, xtrain, ytrain, epochs=100, learning_rate=0.001):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        weight = np.zeros((xtrain.shape[1],1))
        for _ in range(epochs):
            # select samples
            indice = np.random.choice(len(xtrain), len(xtrain))
            x = xtrain[indice,:]
            y = ytrain[indice,:]
            # calculate gradient
            grad = 2/xtrain.shape[0] * xtrain.T.dot(xtrain.dot(weight) - ytrain)
            grad *= learning_rate
            weight -= grad
        return weight

    # =================
    # RIDGE REGRESSION

    def ridge_fit_closed(self, xtrain, ytrain, c_lambda):  # [5pts]
        """
        Args:
            xtrain: N x D numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: N x 1 numpy array, the true labels
            c_lambda: floating number
        Return:
            weight: Dx1 numpy array, the weights of ridge regression model
        """
        tmp = np.linalg.pinv(np.dot(xtrain.T, xtrain) + c_lambda*np.diag(np.ones(xtrain.shape[1])))
        weight = tmp.dot(xtrain.T).dot(ytrain)
        return weight

    def ridge_fit_GD(self, xtrain, ytrain, c_lambda, epochs=500, learning_rate=1e-7):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
            c_lambda: floating number
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        # weights initialization
        weight = np.zeros((xtrain.shape[1],1))
        for _ in range(epochs):
            # calculate gradient
            grad = 2/xtrain.shape[0] * xtrain.T.dot(xtrain.dot(weight) - ytrain) + 2*c_lambda*weight
            grad *= learning_rate
            weight -= grad
        return weight 

    def ridge_fit_SGD(self, xtrain, ytrain, c_lambda, epochs=100, learning_rate=0.001):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        weight = np.zeros((xtrain.shape[1],1))
        for _ in range(epochs):
            # select samples
            indice = np.random.choice(len(xtrain), len(xtrain))
            x = xtrain[indice,:]
            y = ytrain[indice,:]
            # calculate gradient
            grad = 2/xtrain.shape[0] * xtrain.T.dot(xtrain.dot(weight) - ytrain) + 2*c_lambda*weight
            grad *= learning_rate
            weight -= grad
        return weight

    def ridge_cross_validation(self, X, y, kfold=10, c_lambda=100):  # [8 pts]
        """
        Args: 
            X : NxD numpy array, where N is the number of instances and D is the dimensionality of each instance
            y : Nx1 numpy array, true labels
            kfold: Number of folds you should take while implementing cross validation.
            c_lambda: Value of regularization constant
        Returns:
            meanErrors: Float average rmse error
        Hint: np.concatenate might be helpful.
        Look at 3.5 to see how this function is being used.
        # For cross validation, use 10-fold method and only use it for your training data (you already have the train_indices to get training data).
        # For the training data, split them in 10 folds which means that use 10 percent of training data for test and 90 percent for training.
        """
        indice = set(range(len(X)))
        n = len(X) // 10
        kfolds = []
        for i in range(kfold-1):
            indi = np.random.choice(list(indice), n, replace=False)
            kfolds.append(indi)
            indice -= set(indi)
        kfolds.append(np.array(list(indice)))
        errs = []
        for i in range(kfold):
            xtest = X[kfolds[i]]
            ytest = y[kfolds[i]]
            ind = np.concatenate(kfolds[:i]+kfolds[i+1:])
            xtrain = X[ind]
            ytrain = y[ind]
            weight = self.ridge_fit_closed(xtrain, ytrain, c_lambda)
            errs.append(self.rmse(xtest.dot(weight), ytest))
        return np.mean(errs)
        
