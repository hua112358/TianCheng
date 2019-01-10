#!/usr/bin/env python
# -*- coding:utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import model
import time
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

# Get Data
t_get_data_start = time.time()
operation_train, transaction_train, tag_train, operation_test, transaction_test, tag_test = model.GetData("../Data/")
t_get_data_end = time.time()

# Data Exploration
t_data_exploration_start = time.time()
# model.DataExploration(operation_train, tag_train)
# model.DataExploration(transaction_train, tag_train)
# model.DataExploration(operation_test)
# model.DataExploration(transaction_test)
t_data_exploration_end = time.time()

# Data Preprocessing
t_data_preprocessing_start = time.time()
operation_train = model.DataPreprocessing(operation_train)
transaction_train = model.DataPreprocessing(transaction_train)
operation_test = model.DataPreprocessing(operation_test)
transaction_test = model.DataPreprocessing(transaction_test)
t_data_preprocessing_end = time.time()

# Feature Engineering
t_feature_engineering_start = time.time()
x_train, x_test = model.FeatureEngineering(operation_train, transaction_train, tag_train, operation_test, transaction_test, tag_test)
t_feature_engineering_end = time.time()

# Model Optimization
t_model_optimization_start = time.time()
lr_params = [{"C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10]}, 
             {"class_weight": [None, "balanced"]}, 
             {"max_iter": [100, 300, 500, 1000]}, 
             {"penalty": ["l1", "l2"]},
             {"solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}]
lr_best_params = model.ModelOptimization(LogisticRegression(), lr_params, x_train, tag_train["Tag"])

gbdt_params = [{"n_estimators": [100, 300, 500, 1000]}, 
               {"learning_rate": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]}, 
               {"max_features": [None, "log2", "sqrt"]}, 
               {"max_depth": [3, 5, 7, 9]}, 
               {"min_samples_split": [2, 4, 6, 8]}, 
               {"min_samples_leaf": [1, 3, 5, 7]}]
gbdt_best_params = model.ModelOptimization(GradientBoostingClassifier(), gbdt_params, x_train, tag_train["Tag"])

xgb_params = [{"learning_rate": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]}, 
              {"n_estimators": [100, 300, 500, 1000]}, 
              {"max_depth": range(3,10,2)}, 
              {"min_child_weight": range(1,6,2)}, 
              {"gamma": [i/10.0 for i in range(0,5)]}, 
              {"subsample": [i/10.0 for i in range(6,10)]},
              {"colsample_bytree": [i/10.0 for i in range(6,10)]}, 
              {"reg_alpha": [1e-5, 1e-2, 0.1, 1, 100]}]
xgb_best_params = model.ModelOptimization(XGBClassifier(), xgb_params, x_train, tag_train["Tag"])
t_model_optimization_end = time.time()

# Model Evaluation
t_model_evaluation_start = time.time()
lr = LogisticRegression(C = 10, class_weight = "balanced", max_iter = 1000, penalty = "l2", solver = "newton-cg")
lr_roc_auc_score, lr_tpr_weight_score = model.ModelEvaluation(lr, x_train, tag_train["Tag"])
model.Record(x_train, lr, lr_roc_auc_score, lr_tpr_weight_score)

gbdt = GradientBoostingClassifier(n_estimators = 1000, learning_rate = 0.3, max_features = None, 
                                  max_depth = 3, min_samples_split = 4, min_samples_leaf = 7)
gbdt_roc_auc_score, gbdt_tpr_weight_score = model.ModelEvaluation(gbdt, x_train, tag_train["Tag"])
model.Record(x_train, gbdt, gbdt_roc_auc_score, gbdt_tpr_weight_score)

xgb = XGBClassifier(n_estimators = 300, learning_rate = 0.3, max_depth = 5, min_child_weight = 1, 
                    gamma = 0.1, subsample = 0.8, colsample_bytree = 0.8, reg_alpha = 1e-05)
xgb_roc_auc_score, xgb_tpr_weight_score = model.ModelEvaluation(xgb, x_train, tag_train["Tag"])
model.Record(x_train, xgb, xgb_roc_auc_score, xgb_tpr_weight_score)
t_model_evaluation_end = time.time()

# Fit and Predict
t_fit_predict_start = time.time()
lr_y_pred = model.FitPredict(lr, x_train, tag_train["Tag"], x_test)
gbdt_y_pred = model.FitPredict(gbdt, x_train, tag_train["Tag"], x_test)
xgb_y_pred = model.FitPredict(xgb, x_train, tag_train["Tag"], x_test)
t_fit_predict_end = time.time()
print("t_fit_predict: ", t_fit_predict_end - t_fit_predict_start)

# Ensembling
t_ensembling_start = time.time()

y_train = tag_train["Tag"]
kf = KFold(n_splits = 3)
ensembling_roc_auc_scores = []
ensembling_tpr_weight_scores = []
for train_index, test_index in kf.split(x_train):
    x_tr, x_te = x_train.values[train_index], x_train.values[test_index]
    y_tr, y_te = y_train.values[train_index], y_train.values[test_index]
    
    lr_y_pred_train = model.FitPredict(lr, x_tr, y_tr, x_te)
    gbdt_y_pred_train = model.FitPredict(gbdt, x_tr, y_tr, x_te)
    xgb_y_pred_train = model.FitPredict(xgb, x_tr, y_tr, x_te)
    y_pred_list_train = [lr_y_pred_train, gbdt_y_pred_train, xgb_y_pred_train]
    ensembling_y_pred_train = model.Ensembling(y_pred_list_train)
    ensembling_roc_auc_score = roc_auc_score(y_te, ensembling_y_pred_train)
    ensembling_tpr_weight_score = model.tpr_weight_function(y_te, ensembling_y_pred_train)
    ensembling_roc_auc_scores.append(ensembling_roc_auc_score)
    ensembling_tpr_weight_scores.append(ensembling_tpr_weight_score)

ensembling_roc_auc_score = np.mean(ensembling_roc_auc_scores)
ensembling_tpr_weight_score = np.mean(ensembling_tpr_weight_scores)

with open("../Records/record.txt", "a") as f:
    f.write("ensembling\n\n")
    f.write("roc_auc_score:\t")
    f.write(str(ensembling_roc_auc_score))
    f.write("\n\n")
    f.write("tpr_weight_score:\t")
    f.write(str(ensembling_tpr_weight_score))
    f.write("\n")
    f.write("#" * 100)
    f.write("\n")

y_pred_list = [lr_y_pred, gbdt_y_pred, xgb_y_pred]
ensembling_y_pred = model.Ensembling(y_pred_list)
t_ensembling_end = time.time()

# Submit
t_submit_start = time.time()
lr_submission = model.Submit(lr_y_pred, tag_test)
gbdt_submission = model.Submit(gbdt_y_pred, tag_test)
xgb_submission = model.Submit(xgb_y_pred, tag_test)
ensembling_submission = model.Submit(ensembling_y_pred, tag_test)
lr_submission.to_csv("../Submission/lr_submission.csv", index = False)
gbdt_submission.to_csv("../Submission/gbdt_submission.csv", index = False)
xgb_submission.to_csv("../Submission/xgb_submission.csv", index = False)
ensembling_submission.to_csv("../Submission/ensembling_submission.csv", index = False)
t_submit_end = time.time()

print("t_get_data: ", t_get_data_end - t_get_data_start)
print("t_data_preprocessing: ", t_data_preprocessing_end - t_data_preprocessing_start)
print("t_data_exploration: ", t_data_exploration_end - t_data_exploration_start)
print("t_feature_engineering: ", t_feature_engineering_end - t_feature_engineering_start)
print("t_model_optimization: ", t_model_optimization_end - t_model_optimization_start)
print("t_model_evaluation: ", t_model_evaluation_end - t_model_evaluation_start)
print("t_ensembling: ", t_ensembling_end - t_ensembling_start)
print("t_submit: ", t_submit_end - t_submit_start)
print("total_time: ", t_submit_end - t_get_data_start)