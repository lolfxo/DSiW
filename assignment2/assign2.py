# template for assign2.py

import numpy as np
import pandas as pd
import random as rd
from sklearn import preprocessing as spp
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder



def read_data(data_path, train_path, validation_path, test_path):
    # train_data, train_target = np.zeros((1000000, 39)), np.zeros((1000000,))
    # validation_data, validation_target = np.zeros((250000, 39)), np.zeros((250000,))
    # test_data, test_target = np.zeros((750000, 39)), np.zeros((750000,))

    ### READ INDICES
    #num_lines = sum(1 for line in open('train.txt'))
    num_lines = sum(1 for line in open(data_path))
    all_ind = np.arange(0,int(num_lines))
    set_all_ind = set(all_ind)
    
    # read in row numbers of train, validation and test data to be read
    with open(train_path, 'r') as myfile:
        train_ind = myfile.read().split(',')
    
    train_ind = list(map(int, train_ind))
        
    with open(validation_path, 'r') as myfile:
        valid_ind = myfile.read().split(',')
        
    valid_ind = list(map(int, valid_ind))
    
    with open(test_path, 'r') as myfile:
        test_ind = myfile.read().split(',')
    
    test_ind = list(map(int, test_ind))
    
    ### GENERATE INDICES
    # generate TRAIN indices to skip
    to_skip_train = list(set_all_ind.difference(train_ind))
    train1M = pd.read_csv(data_path, skiprows = to_skip_train, sep='\t', header=None)

    # generate VALIDATION indices to skip
    to_skip_valid = list(set_all_ind.difference(valid_ind))
    validation250k = pd.read_csv(data_path, skiprows = to_skip_valid, sep='\t', header=None)

    # generate TEST indices to skip
    to_skip_test = list(set_all_ind.difference(test_ind))
    test750k = pd.read_csv(data_path, skiprows = to_skip_test, sep='\t', header=None)
    
    ### SEPARATE TARGET AND DATA
    train_target = train1M.iloc[:,0]
    train_data = train1M.iloc[:,1:40]
    
    validation_target = validation250k.iloc[:,0]
    validation_data = validation250k.iloc[:,1:40]
    
    test_target = test750k.iloc[:,0]
    test_data = test750k.iloc[:,1:40]
    
    return train_data, train_target, validation_data, validation_target, test_data, test_target


def preprocess_int_data(data, features):
    n = len([f for f in features if f < 13])
    int_data = data.iloc[:,0:13]
    
    int_data = int_data.fillna(0)
    int_data = int_data.replace(-1,0)
    int_data = int_data.replace(-2,0)
    int_data = int_data.replace(-3,0)
    
    scaler = spp.StandardScaler()
    scaler.fit(int_data)
    scaled_int_data = scaler.transform(int_data)
    
    return scaled_int_data


def preprocess_cat_data(data, features, preprocess):
    data.iloc[:,13:] = data.iloc[:,13:].fillna(0)
    
    n = 19
    
    feature_array = np.array(features)
    # pick out the categorical features only
    feature_array = feature_array[feature_array>12]
    
    for f in feature_array:
        cat_vc = data.iloc[:,f].value_counts()
        for i in range(0, len(cat_vc)):
            if (i > n):
                data.iloc[:,f] = data.iloc[:,f].replace(cat_vc.index[i],'Other')
    
    cat_data = data.iloc[:,feature_array]

    if (preprocess == 'onehot'):
        lencoder = LabelEncoder()
        cat_data_enc = cat_data.apply(lencoder.fit_transform)
    
        ohe_dict = {}
        cat_data_dict = {}

        for i in feature_array:
            ohe = OneHotEncoder()
            ohe.fit_transform(cat_data_enc[i+1].values.reshape(-1,1))
            ohe_dict[i] = ohe.transform(cat_data_enc[i+1].values.reshape(-1,1))

    # needs to return proper categorical data.
    return ohe_dict
