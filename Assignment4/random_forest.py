import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

class RandomForest(object):
    def __init__(self, n_estimators=50, max_depth=None, max_features=0.7):
        # helper function. You don't have to modify it
        # Initialization done here
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstraps_row_indices = []
        self.feature_indices = []
        self.out_of_bag = []
        self.decision_trees = [sklearn.tree.DecisionTreeClassifier(max_depth=max_depth, criterion='entropy') for i in range(n_estimators)]
        
    def _bootstrapping(self, num_training, num_features, random_seed = None):
        """
        TODO: 
        - Randomly select a sample dataset of size num_training with replacement from the original dataset. 
        - Randomly select certain number of features (num_features denotes the total number of features in X, 
          max_features denotes the percentage of features that are used to fit each decision tree) without replacement from the total number of features.
        
        Return:
        - row_idx: the row indices corresponding to the row locations of the selected samples in the original dataset.
        - col_idx: the column indices corresponding to the column locations of the selected features in the original feature list.
        
        Reference: https://en.wikipedia.org/wiki/Bootstrapping_(statistics)
        
        Hint: Consider using np.random.choice.
        """
        np.random.seed(random_seed)
        row_idx = np.random.choice(num_training, size= num_training, replace=True)
        col_idx = np.random.choice(num_features, size= int(num_features*self.max_features), replace=False)

        
        return  row_idx, col_idx
            
    def bootstrapping(self, num_training, num_features):
        # helper function. You don't have to modify it
        # Initializing the bootstap datasets for each tree
        for i in range(self.n_estimators):
            total = set(list(range(num_training)))
            row_idx, col_idx = self._bootstrapping(num_training, num_features)
            total = total - set(row_idx)
            self.bootstraps_row_indices.append(row_idx)
            self.feature_indices.append(col_idx)
            self.out_of_bag.append(total)

    def fit(self, X, y):
        """
        TODO:
        Train decision trees using the bootstrapped datasets.
        Note that you need to use the row indices and column indices.
        
        X: NxD numpy array, where N is number 
           of instances and D is the dimensionality of each 
           instance
        y: Nx1 numpy array, the predicted labels
        """
        num_training, num_features = X.shape
        self.bootstrapping(num_training, num_features)
        for i in range(self.n_estimators):
            sample_x = X[self.bootstraps_row_indices[i]]
            sample_y = y[self.bootstraps_row_indices[i]]
            sample_x = sample_x[:,self.feature_indices[i]]
            self.decision_trees[i].fit(sample_x, sample_y)
        return self.decision_trees

    def OOB_score(self, X, y):
        # helper function. You don't have to modify it
        # This function computes the accuracy of the random forest model predicting y given x.
        accuracy = []
        for i in range(len(X)):
            predictions = []
            for t in range(self.n_estimators):
                if i in self.out_of_bag[t]:
                    predictions.append(self.decision_trees[t].predict(np.reshape(X[i][self.feature_indices[t]], (1,-1)))[0])
            if len(predictions) > 0:
                accuracy.append(np.sum(predictions == y[i]) / float(len(predictions)))
        return np.mean(accuracy)

    
    def plot_feature_importance(self, random_forest):
        """
        TODO:
        -Display a bar plot showing the feature importance of every feature in 
        at least one decision tree from the tuned random_forest from Q3.2.
        
        Args:
            data_train: This is the orginal data train Dataframe containg data AND labels.
                Hint: you can access labels with data_train.columns
        Returns:
            None. Calling this function should simply display the aforementioned feature importance bar chart
        """
        n = 5  # how many decision trees you want to display, n < self.n_estimators
        if n > self.n_estimators:
            n = self.n_estimators
        data_train = random_forest
        X_train = data_train.drop(columns='num')
        y_train = data_train['num']
        y_train = y_train.to_numpy()
        y_train[y_train > 1] = 1
        X_train, y_train, = np.array(X_train), np.array(y_train)
        self.fit(X_train, y_train)
        for i in range(n):
            importances = self.decision_trees[i].feature_importances_
            indices = np.argsort(importances)[::-1]
            X = data_train.columns[[indices]]
            Y = importances[[indices]]
            plt.bar(X, Y, 0.4, color="green")
            plt.xlabel("Feature")
            plt.ylabel("Importance")
            plt.title("Feature importance for DT: " + str(i))
            plt.show()
            # for f in range(len(indices)):
            #     print("%2d) %-*s %f" % (f + 1, 30, data_train.columns[indices[f]], importances[indices[f]]))
