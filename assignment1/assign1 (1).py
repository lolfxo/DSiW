import argparse

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_rcv1

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import BernoulliNB

def load_labels(path_to_labels):
    labels = pd.read_csv(path_to_labels, names=['label'], dtype=np.int32)
    return labels['label'].tolist()


def load_training_data():
    data = fetch_rcv1(subset='train')
    return data.data, data.target.toarray(), data.sample_id


def load_validation_data(path_to_ids):
    data = fetch_rcv1(subset='test')
    ids = pd.read_csv(path_to_ids, names=['id'], dtype=np.int32)
    mask = np.isin(data.sample_id, ids['id'])
    validation_data = data.data[mask]
    validation_target = data.target[mask].toarray()
    validation_ids = data.sample_id[mask]
    return validation_data, validation_target, validation_ids


class CS5304BaseClassifier(object):
    def __init__(self):
        pass

    def train(self, x, y):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class CS5304KNNClassifier(CS5304BaseClassifier):
    """
    http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    """ 
    def __init__(self, n_neighbors):
        self.k = n_neighbors # input variable name to class: n_neighbors
        self.knn = KNeighborsClassifier(n_neighbors = self.k) # input variable name to method: KNeighborsClassifier

    def train(self, train_data, train_target):
        # train KNN classifier
        self.knn.fit(train_data, train_target.ravel())
        
    def predict(self, eval_data):
        # eval_data should have the same number of features (columns) as train_data
        output = self.knn.predict(eval_data)
        return output


class CS5304NBClassifier(CS5304BaseClassifier):
    """
    http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html
    """
    def __init__(self):
        self.nb_clf = BernoulliNB()
    
    def train(self, train_data, train_target):
        # train NB classifier
        self.nb_clf.fit(train_data, train_target.ravel())
        
    def predict(self, eval_data):
        # eval_data should have the same number of features (columns) as train_data
        output = self.nb_clf.predict(eval_data)
        return output
        
        

class CS5304KMeansClassifier(CS5304BaseClassifier):
    """
    http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    """
    def __init__(self, n_clusters=2, init='k-means++'):
        self.n_clstrs = n_clusters
        self.init1 = init
        self.kmeans = KMeans(n_clusters=self.n_clstrs, init = self.init1)
  
    def train(self, train_data, train_target):
        # find initial centroids:
        
        # train KMeans clustering:
        self.kmeans.fit(train_data)
    
    def predict(self, eval_data):
        output = self.kmeans.predict(eval_data)
        return output
    
    
if __name__ == '__main__':

    # This is an example of loading the training and validation data. You may use this snippet
    # when completing the exercises for the assignment.

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_labels", default="labels.txt")
    parser.add_argument("--path_to_ids", default="validation.txt")
    options = parser.parse_args()

    labels = load_labels(options.path_to_labels)
    train_data, train_target, _ = load_training_data()
    eval_data, eval_target, _ = load_validation_data(options.path_to_ids)
