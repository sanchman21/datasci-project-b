'''
This script is used to train a single XGBoost classifier using only clinical features.
'''

# import libraries
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import metrics
import gc 

# load the data
df = pd.read_csv("../datasets/patients_fold.csv")
rechecked_patient_ids = [2209722160, 2209801259, 2209801421, 2209802027, 2209802125] # rechecked patient ids
df = df.loc[df["patient_id"] != 2209801848] # remove the patient with missing data
df.loc[df["patient_id"].isin(rechecked_patient_ids), "morphology"] = 0 # set the morphology of rechecked patients to 0

def train_single_sklearn(x, y, num_class, seed, feature_names, booster):
    """
    Train a single xgb classifier
    """
    dmat = xgb.DMatrix(x, y, feature_names=feature_names) # create the DMatrix

    # define the objective and evaluation metric based on the number of classes (problem is binary)
    if num_class > 2:
        metric = 'mlogloss'
        objective = 'multi:softprob'
    elif num_class == 2:
        metric = 'logloss'
        objective = 'binary:logistic'
    else:
        raise RuntimeError(f'num_class = {num_class}, must be >= 2')

    random_forest = False

    # define the parameters based on the booster
    if random_forest: # params if random forest is used
        params = dict(
            objective=objective,
            eval_metric=metric,
            nthread=4,
            colsample_bynode=0.6,
            learning_rate=1,
            max_depth=5,
            num_parallel_tree=100,
            subsample=0.6,
        )
        model = xgb.train(params, dmat, num_boost_round=1) # train the model
    else: # params if random forest is not used
        params = dict( # params if random forest is not used
            booster=booster,
            objective=objective,
            eval_metric=metric,
            nthread=4,
        )
        if num_class > 2: # if the number of classes is greater than 2
            params['num_class'] = num_class
        
        # define the cross-validation parameters
        xgb_cv_params = dict(
            metrics=metric,
            num_boost_round=200,
            early_stopping_rounds=10,
            stratified=True,
            nfold=10,
            seed=seed,
        )

        # find best num_boost_rounds for the optimal parameters
        # cv = xgb.cv(params, dmat, **xgb_cv_params)
        # num_boost_round = cv[f'test-{metric}-mean'].idxmin()

        # model = xgb.train(params, dmat, num_boost_round=num_boost_round)
        # params['num_boost_round'] = num_boost_round
        model = xgb.XGBClassifier(**params, importance_type='gain', validate_parameters=True) # create the model
        df_x = pd.DataFrame(x) # create a dataframe of the features
        df_x.columns = feature_names # set the column names
        model.fit(df_x, y) # fit the model

    return model, params # return the model and the parameters

feature_columns = ['Age', 'Gender', 'Haemoglobin',
    'MCV', 'White cell count', 'Neutrophil count', 'Monocyte count',
    'Platelet count', 'Blast percentage (PB)', 'LDH'
] # clinical features
target_column = 'morphology' # target column
booster = 'gbtree' # booster

accuracies = [] # list to store the accuracies
aurocs = [] # list to store the AUROCs

for i in range(5): # iterate over the 5 folds
    set_col = f'set{i}' # set column
    x_values = df[df[set_col] == 'train'][feature_columns].values # features
    y_values = df[df[set_col] == 'train'][target_column].values # target
    # train the model
    model, params = train_single_sklearn(x_values, y_values, num_class=2, seed=123, feature_names=feature_columns, booster=booster)
    X_test = df[df[set_col] == 'test'][feature_columns].values # test features
    y_test = df[df[set_col] == 'test'][target_column].values # test target
    # if booster == 'gbtree':
    #      preds = model.predict(X_test, iteration_range=(0, model.best_iteration + 1))
    # else:
    preds = model.predict(X_test) # predict the target
    preds = preds.astype(int) # convert the predictions to integers
    accuracy = accuracy_score(y_test, preds) # calculate the accuracy
    auroc = roc_auc_score(y_test, preds) # calculate the AUROC
    accuracies.append(accuracy) # append the accuracy to the list
    aurocs.append(auroc) # append the AUROC to the list
    print(f"Accuracy: {round(accuracy*100, 2)}, AUROC: {round(auroc*100, 2)}")
    model._Booster.__del__() # delete the booster
    del model # delete the model
    gc.collect() # collect the garbage
    
print(f"Accuracy: {round(np.mean(accuracies)*100, 2)} +- {round(np.std(accuracies)*100, 2)}")
print(f"AUROC: {round(np.mean(aurocs)*100, 2)} +- {round(np.std(aurocs)*100, 2)}")
