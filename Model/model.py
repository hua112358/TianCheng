#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: hua112358

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time 

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

"""
    Step 0 : Get Data
    
"""

def GetData(data_path):
    print("Get Data Start")
    operation_train = pd.read_csv(data_path + "operation_train_new.csv")
    transaction_train = pd.read_csv(data_path + "transaction_train_new.csv")
    operation_test = pd.read_csv(data_path + "operation_round1_new.csv")
    transaction_test = pd.read_csv(data_path + "transaction_round1_new.csv")
    tag_train = pd.read_csv(data_path + "tag_train_new.csv")
    tag_test = pd.read_csv("../Data/submission_example.csv")
    print("Get Data Done")
    return operation_train, transaction_train, tag_train, operation_test, transaction_test, tag_test

"""
    Step 1 : Data Exploration
    
    Categorical Columns : countplot, barplot
    Numerical Columns : regplot
    
"""

def DataExploration(data, tag = None):
    print("Data Exploration Start")
    plt.figure()
    plt.title("UID_count")
    uids = np.sort(data["UID"].unique())
    count = data.groupby("UID")["time"].agg(["count"])["count"]
    sns.regplot(x = uids, y = count)
    plt.show()
#     for column in data.columns:
#         if column != "UID" and data[column].nunique() < 50:
#             plt.figure()
#             sns.countplot(x = column, data = data)
#             plt.show()

    if tag is not None:
        uid_tag_dict = dict(zip(tag["UID"], tag["Tag"]))
        data["tag"] = data["UID"].map(uid_tag_dict)
        for column in data.columns:
            if data[column].unique().shape[0] < 50 and column != "tag":
                plt.figure()
                sns.barplot(x = column, y = "tag", data = data.sample(100000))
                plt.show()
        
    print("Data Exploration Done")
    
""" 
    Step 2 : Data Preprocessing 

    Drop Duplicates
    Drop Outliers: After Data Exploration 
    
"""

def DataPreprocessing(data):
    print("Data Preprocessing Start")
    data = data[data["UID"] != 17520]
    data = data.drop_duplicates()
    print("Data Preprocessing Done")
    return data

"""
    Step 3 : Feature Engineering
    
    Feature Creation
    Feature Selection
    
"""

def FeatureCreation(data, tag):
    print("Feature Creation Start")
    data["hour"] = data["time"].apply(lambda x : int(x[:2]))
    features = pd.DataFrame(tag["UID"])
    
    # Feature Creation 1: column_nunique
    for column in data.columns:
        if column != "UID":
            print("Create " + column + "_nunique...")
            column_nunique = data.groupby("UID")[column].agg(["nunique"]).reset_index().rename(columns = {"nunique": column + "_nunique"})
            features = features.merge(column_nunique, on = "UID", how = "left")

    # Feature Creation 2: column_nunique_UID and column_count_UID
    data_copy = data.copy()
    for column in data.columns: 
        if column != "UID":
            print("Create " + column + "_nunique_UID and " + column + "_count_UID...")
            column_nunique_count = data_copy.groupby(column)["UID"].agg(["nunique", "count"]).reset_index().rename(columns = {"nunique": column + "_nunique_UID", "count": column + "_count_UID"})
            data_copy = data_copy.merge(column_nunique_count, on = column, how = "left")

    column_nunique = [col + "_nunique_UID" for col in data.columns if col != "UID"]
    column_count = [col + "_count_UID" for col in data.columns if col != "UID"]
    columns = column_nunique + column_count
    
    for column in columns:
        print("Create " + column + "...")
        column_nunique_count_UID = data_copy.groupby("UID")[column].agg(["max", "min", "mean"]).reset_index().rename(columns = {"max": column + "_max", "min": column + "_min", "mean": column + "_mean"})
        features = features.merge(column_nunique_count_UID, on = "UID", how = "left")
    
    # Feature Creation 3: day_frequency, hour_frequency
    day_frequency = data.groupby(["UID", "day"])["time"].agg(["count"]).reset_index().groupby("UID")["count"].agg(["max", "min", "mean"]).rename(columns = {"max": "day_frequency_max", "min": "day_frequency_min", "mean": "day_frequency_mean"})
    features = features.merge(day_frequency, on = "UID", how = "left")
    hour_frequency = data.groupby(["UID", "day", "hour"])["time"].agg(["count"]).reset_index().groupby("UID")["count"].agg(["max", "min", "mean"]).rename(columns = {"max": "hour_frequency_max", "min": "hour_frequency_min", "mean": "hour_frequency_mean"})
    features = features.merge(hour_frequency, on = "UID", how = "left")

    print("Feature Creation Done")
    
    return features

def FeatureSelection(x_train, y_train, x_test):
    print("Feature Selection Start")
    sfm = SelectFromModel(GradientBoostingClassifier())
    sfm.fit(x_train, y_train)
    support = sfm.get_support()
    indices = list(range(len(support)))
    selected_indices = [index for index in indices if support[index]]
    selected_features = x_train.columns.values[selected_indices]
    x_train = x_train.loc[:, selected_features]
    x_temp = pd.DataFrame(columns = selected_features)
    for feature in selected_features:
        if feature in x_test.columns:
            x_temp[feature] = x_test[feature]
    x_test = x_temp
    print("Feature Selection Done")
    return x_train, x_test

def FeatureEngineering(operation_train, transaction_train, tag_train, operation_test, transaction_test, tag_test):
    print("Feature Engineering Start")
    operation_train_features = FeatureCreation(operation_train, tag_train)
    transaction_train_features = FeatureCreation(transaction_train, tag_train)
    operation_test_features = FeatureCreation(operation_test, tag_test)
    transaction_test_features = FeatureCreation(transaction_test, tag_test)
    x_train = operation_train_features.merge(transaction_train_features, on = "UID", how = "left")
    y_train = tag_train["Tag"]
    x_test = operation_test_features.merge(transaction_test_features, on = "UID", how = "left")
    
    # Feature Creation 3: success_mean, os_has_105, os_has_107, channel_has_118, channel_has_119
    success_mean = operation_train.groupby("UID")["success"].agg(["mean"]).reset_index().rename(columns = {"mean": "success_mean"})
    x_train = x_train.merge(success_mean, on = "UID", how = "left")
    
    os_has_105 = operation_train.groupby("UID")["os"].agg(lambda x : 105 in x.values).reset_index().rename(columns = {"os": "os_has_105"})
    os_has_105["os_has_105"] = os_has_105["os_has_105"].apply(int)
    x_train = x_train.merge(os_has_105, on = "UID", how = "left")
    
    os_has_107 = operation_train.groupby("UID")["os"].agg(lambda x : 107 in x.values).reset_index().rename(columns = {"os": "os_has_107"})
    os_has_107["os_has_107"] = os_has_107["os_has_107"].apply(int)
    x_trian = x_train.merge(os_has_107, on = "UID", how = "left")
    
    channel_has_118 = transaction_train.groupby("UID")["channel"].agg(lambda x : 118 in x.values).reset_index().rename(columns = {"channel": "channel_has_118"})
    channel_has_118["channel_has_118"] = channel_has_118["channel_has_118"].apply(int)
    x_train = x_train.merge(channel_has_118, on = "UID", how = "left")
    
    channel_has_119 = transaction_train.groupby("UID")["channel"].agg(lambda x : 119 in x.values).reset_index().rename(columns = {"channel": "channel_has_119"})
    channel_has_119["channel_has_119"] = channel_has_119["channel_has_119"].apply(int)
    x_train = x_train.merge(channel_has_119, on = "UID", how = "left")
    
    success_mean = operation_test.groupby("UID")["success"].agg(["mean"]).reset_index().rename(columns = {"mean": "success_mean"})
    x_test = x_test.merge(success_mean, on = "UID", how = "left")
    
    os_has_105 = operation_test.groupby("UID")["os"].agg(lambda x : 105 in x.values).reset_index().rename(columns = {"os": "os_has_105"})
    os_has_105["os_has_105"] = os_has_105["os_has_105"].apply(int)
    x_test = x_test.merge(os_has_105, on = "UID", how = "left")
    
    os_has_107 = operation_test.groupby("UID")["os"].agg(lambda x : 107 in x.values).reset_index().rename(columns = {"os": "os_has_107"})
    os_has_107["os_has_107"] = os_has_107["os_has_107"].apply(int)
    x_trian = x_test.merge(os_has_107, on = "UID", how = "left")
    
    channel_has_118 = transaction_test.groupby("UID")["channel"].agg(lambda x : 118 in x.values).reset_index().rename(columns = {"channel": "channel_has_118"})
    channel_has_118["channel_has_118"] = channel_has_118["channel_has_118"].apply(int)
    x_test = x_test.merge(channel_has_118, on = "UID", how = "left")
    
    channel_has_119 = transaction_test.groupby("UID")["channel"].agg(lambda x : 119 in x.values).reset_index().rename(columns = {"channel": "channel_has_119"})
    channel_has_119["channel_has_119"] = channel_has_119["channel_has_119"].apply(int)
    x_test = x_test.merge(channel_has_119, on = "UID", how = "left")
    
    # Feature Selection
    x_train = x_train.fillna(-1)
    x_test = x_test.fillna(-1)
    x_train, x_test = FeatureSelection(x_train, y_train, x_test)
    print("Feature Engineering Done")
    return x_train, x_test
    
"""
    Step 4 : Model Optimization
    
    Models : lr, gbdt, xgb
    
"""

def ModelOptimization(model, params, x_train, y_train):
    print("Model Optimizatioin Start")
    x_train = x_train.fillna(-1)
    best_params = []
#     cv = GridSearchCV(estimator = model, param_grid = params, scoring = "roc_auc", cv = 3, n_jobs = -1)
#     cv.fit(x_train, y_train)
    for param in params:
        print("Optimize param", param, "...")
        cv = GridSearchCV(estimator = model, param_grid = param, scoring = "roc_auc", cv = 3, n_jobs = -1)
        cv.fit(x_train, y_train)
        best_params.append(cv.best_params_)
    print("Model Optimizatioin Done")
    return best_params
    
"""
    Step 5 : Model Evaluation
    
"""

def tpr_weight_function(y_true, y_predict):
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    return 0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3

def ModelEvaluation(model, x_train, y_train):
    print("Model Evaluation Start")
    
    # roc_auc
    print("Compute roc_auc_score...")
    roc_auc = np.mean(cross_val_score(estimator = model, 
                                         X = x_train, 
                                         y = y_train, 
                                         scoring = "roc_auc", 
                                         cv = 3, 
                                         n_jobs = -1, 
                                         verbose = 10))

    # tpr_weight
    print("Compute tpr_weight_score...")
    kf = KFold(n_splits = 3)
    model_scores = []
    for train_index, test_index in kf.split(x_train):
        print("Split data...")
        x_tr, x_te = x_train.values[train_index], x_train.values[test_index]
        y_tr, y_te = y_train.values[train_index], y_train.values[test_index]
        model.fit(x_tr, y_tr)
        y_pred = model.predict(x_te)
        score = tpr_weight_function(y_te, y_pred)
        model_scores.append(score)
    tpr_weight = np.mean(model_scores)
    print("Model Evaluation Done")
    return roc_auc, tpr_weight

def Record(x_train, model, roc_auc_score, tpr_weight_score):
    print("Record Start")
    with open("../Records/record.txt", "a") as f:
        f.write("features:\t")
        f.write("[" + ", ".join(x_train.columns.values) + "]")
        f.write("\n\n")
        f.write("model:\t")
        f.write(str(model))
        f.write("\n\n")
        f.write("roc_auc_score:\t")
        f.write(str(roc_auc_score))
        f.write("\n\n")
        f.write("tpr_weight_score:\t")
        f.write(str(tpr_weight_score))
        f.write("\n")
        f.write("#" * 100)
        f.write("\n")
    print("Record Done")
    
"""
    Step 6 : Fit and Predict
    
"""

def FitPredict(model, x_train, y_train, x_test):
    print("Fit Predict Start")
    model.fit(x_train, y_train)
    y_pred = model.predict_proba(x_test)[:, 1]
    print("Fit Predict Done")
    return y_pred
    
"""
    Step 7 : Ensembling

"""

def Ensembling(y_pred_list):
    print("Ensembling Start")
    ensembling_y_pred = np.array(y_pred_list).mean(axis = 0)
    print("Ensembling Done")
    return ensembling_y_pred
   
"""
    Step 8 : Submit
    
"""
def Submit(y_pred, tag_test):
    print("Submit Start")
    submission = pd.DataFrame({"UID": tag_test["UID"], "Tag": y_pred})
    print("Submit Done")
    return submission