import pandas as pd
import numpy as np
import xgboost as xgb
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import metrics

df = pd.read_csv("../datasets/patients_fold.csv")
rechecked_patient_ids = [2209722160, 2209801259, 2209801421, 2209802027, 2209802125]
df = df.loc[df["patient_id"] != 2209801848]
df.loc[df["patient_id"].isin(rechecked_patient_ids), "morphology"] = 0


def train_single_sklearn(x, y, num_class, seed, feature_names, booster):
    """
    Train a single xgb classifier
    """

    dmat = xgb.DMatrix(x, y, feature_names=feature_names)

    if num_class > 2:
        metric = 'mlogloss'
        objective = 'multi:softprob'
    elif num_class == 2:
        metric = 'logloss'
        objective = 'binary:logistic'
    else:
        raise RuntimeError(f'num_class = {num_class}, must be >= 2')

    # random_forest = True
    random_forest = False

    if random_forest:
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

        model = xgb.train(params, dmat, num_boost_round=1)
    else:
        params = dict(
            booster=booster,
            objective=objective,
            eval_metric=metric,
            nthread=4,
        )
        if num_class > 2:
            params['num_class'] = num_class

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
        model = xgb.XGBClassifier(**params, importance_type='gain', validate_parameters=True)
        df_x = pd.DataFrame(x)
        df_x.columns = feature_names
        model.fit(df_x, y)

    return model, params

feature_columns = ['Age', 'Gender', 'Haemoglobin',
    'MCV', 'White cell count', 'Neutrophil count', 'Monocyte count',
    'Platelet count', 'Blast percentage (PB)', 'LDH'
]
target_column = 'morphology'
booster = 'gbtree'

accuracies = []
aurocs = []

for i in range(5):
    set_col = f'set{i}'
    x_values = df[df[set_col] == 'train'][feature_columns].values
    y_values = df[df[set_col] == 'train'][target_column].values  
    model, params = train_single_sklearn(x_values, y_values, num_class=2, seed=123, feature_names=feature_columns, booster=booster)
    X_test = df[df[set_col] == 'test'][feature_columns].values
    y_test = df[df[set_col] == 'test'][target_column].values
    # if booster == 'gbtree':
    #      preds = model.predict(X_test, iteration_range=(0, model.best_iteration + 1))
    # else:
    preds = model.predict(X_test)
    preds = preds.astype(int)
    accuracy = accuracy_score(y_test, preds)
    auroc = roc_auc_score(y_test, preds)
    accuracies.append(accuracy)
    aurocs.append(auroc)
    print(f"Accuracy: {round(accuracy*100, 2)}, AUROC: {round(auroc*100, 2)}")
    model._Booster.__del__()
    del model    
    import gc
    gc.collect()
    # break
    
print(f"Accuracy: {round(np.mean(accuracies)*100, 2)} +- {round(np.std(accuracies)*100, 2)}")
print(f"AUROC: {round(np.mean(aurocs)*100, 2)} +- {round(np.std(aurocs)*100, 2)}")
