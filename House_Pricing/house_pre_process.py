#!/usr/bin/env python
# coding: utf-8

import joblib
import pandas as pd
import numpy as np
import json
import os

from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

input_path = os.getcwd()

def opening_json():
    json_path = os.path.join(input_path,'data_file.json')
    with open(json_path,'r') as file:
        train_data = json.load(file)
    return train_data

def opening_unique_file():
    unique_path = os.path.join(input_path,'unique_values.csv')
    unique_vals = pd.read_csv(unique_path)
    unique_vals = unique_vals.fillna(-1)
    change = {
        'MSSubClass' : np.int64,
        'MoSold' : np.int64,
        'YrSold': np.int64,
    }
    unique_vals = unique_vals.astype(change)
    unique_vals = unique_vals.replace({-1: None})
    return unique_vals

def handle_missing(features):
    
    #import data from JSON file
    train_data = opening_json()
    missing_train = train_data['missing_values']
    normalize = train_data['normalized_features']
    obj_cols = train_data['obj_cols']
    num_cols = train_data['num_cols']
    
    # Some of the non-numeric predictors are stored as numbers; convert them into strings 
    features['MSSubClass'] = features['MSSubClass'].apply(str)
    features['YrSold'] = features['YrSold'].astype(str)
    features['MoSold'] = features['MoSold'].astype(str)
    
    # the data description states that NA refers to typical ('Typ') values
    features['Functional'] = features['Functional'].fillna('Typ')
    # Replace the missing values in each of the columns below with their mode
    features['Electrical'] = features['Electrical'].fillna("SBrkr")
    features['KitchenQual'] = features['KitchenQual'].fillna("TA")
    features['Exterior1st'] = missing_train['Exterior1st']
    features['Exterior2nd'] = missing_train['Exterior2nd']
    features['SaleType'] = missing_train['SaleType']
    
    #Replacing the missing value of MSZoning based on MSSublass.
    if (features['MSZoning'].isnull()).bool():
        for (key,value) in missing_train["MSZoning"].items(): 
            if (features['MSSubClass'] == key).bool():
                features['MSZoning'] = value
        
    # the data description stats that NA refers to "No Pool"
    #features["PoolQC"] = features["PoolQC"].fillna("None")
    # Replacing the missing values with 0, since no garage = no cars in garage
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        features[col] = features[col].fillna(0)
    # Replacing the missing values with None
    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        features[col] = features[col].fillna('None')
    # NaN values for these categorical basement features, means there's no basement
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        features[col] = features[col].fillna('None')
        
    # Group the by neighborhoods, and fill in missing value by the median LotFrontage of the neighborhood
    if (features['LotFrontage'].isnull()).bool():
        for (key,value) in missing_train["LotFrontage"].items():
            if (features['Neighborhood'] == key).bool():
                features['LotFrontage'] = value
    
    # We have no particular intuition around how to fill in the rest of the categorical features
    # So we replace their missing values with None
    features.update(features[obj_cols].fillna('None'))
        
    # And we do the same thing for numerical features, but this time with 0s
    features.update(features[num_cols].fillna(0))
    
    for (key,value) in normalize.items():
        features[key] = boxcox1p(features[key], value)
    return features


#Creating new features
def create_new_features(test):
    test['BsmtFinType1_Unf'] = 1*(test['BsmtFinType1'] == 'Unf')
    test['HasWoodDeck'] = (test['WoodDeckSF'] == 0) * 1
    test['HasOpenPorch'] = (test['OpenPorchSF'] == 0) * 1
    test['HasEnclosedPorch'] = (test['EnclosedPorch'] == 0) * 1
    test['Has3SsnPorch'] = (test['3SsnPorch'] == 0) * 1
    test['HasScreenPorch'] = (test['ScreenPorch'] == 0) * 1
    test['YearsSinceRemodel'] = test['YrSold'].astype(int) - test['YearRemodAdd'].astype(int)
    test['Total_Home_Quality'] = test['OverallQual'] + test['OverallCond']
    test = test.drop(['Utilities', 'Street', 'PoolQC',], axis=1)
    test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']
    test['YrBltAndRemod'] = test['YearBuilt'] + test['YearRemodAdd']
    
    test['Total_sqr_footage'] = (test['BsmtFinSF1'] + test['BsmtFinSF2'] +
                                     test['1stFlrSF'] + test['2ndFlrSF'])
    test['Total_Bathrooms'] = (test['FullBath'] + (0.5 * test['HalfBath']) +
                                   test['BsmtFullBath'] + (0.5 * test['BsmtHalfBath']))
    test['Total_porch_sf'] = (test['OpenPorchSF'] + test['3SsnPorch'] +
                                  test['EnclosedPorch'] + test['ScreenPorch'] +
                                  test['WoodDeckSF'])
    test['TotalBsmtSF'] = test['TotalBsmtSF'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
    test['2ndFlrSF'] = test['2ndFlrSF'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
    test['GarageArea'] = test['GarageArea'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
    test['GarageCars'] = test['GarageCars'].apply(lambda x: 0 if x <= 0.0 else x)
    test['LotFrontage'] = test['LotFrontage'].apply(lambda x: np.exp(4.2) if x <= 0.0 else x)
    test['MasVnrArea'] = test['MasVnrArea'].apply(lambda x: np.exp(4) if x <= 0.0 else x)
    test['BsmtFinSF1'] = test['BsmtFinSF1'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
    
    test['haspool'] = test['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    test['has2ndfloor'] = test['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    test['hasgarage'] = test['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    test['hasbsmt'] = test['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    test['hasfireplace'] = test['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
    return test


#Feature Transformations
def logs(res):
    ls = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
                 'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',
                 'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
                 'TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF',
                 'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','YearRemodAdd','TotalSF']
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(np.log(1.01+res[l])).values)   
        res.columns.values[m] = l + '_log'
        m += 1
    return res


def squares(res):
    ls = ['YearRemodAdd', 'LotFrontage_log', 
              'TotalBsmtSF_log', '1stFlrSF_log', '2ndFlrSF_log', 'GrLivArea_log',
              'GarageCars_log', 'GarageArea_log']

    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(res[l]*res[l]).values)   
        res.columns.values[m] = l + '_sq'
        m += 1
    return res


#one hot encoding
def one_hot_encoding(test):
    #getting unique values
    unique_vals = opening_unique_file()
    
    new_test = test.append(unique_vals, ignore_index=True, sort=False)
    
    test = pd.get_dummies(new_test).reset_index(drop=True)
    test = test.loc[:,~test.columns.duplicated()]
    
    return test[:1]


#master Function

def master_func(test):
                
    test = pd.DataFrame(test,index = [0])
    test = handle_missing(test)
    test = create_new_features(test)
    test = logs(test)
    test = squares(test)
    test = one_hot_encoding(test)
    
    return test