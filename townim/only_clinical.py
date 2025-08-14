import pandas as pd
import numpy as np
import xgboost as xgb
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
import mlflow
import subprocess
import gc
from utils import save_metrics_csv, plot_save_roc_curve, plot_confusion_matrix

mlflow.set_tracking_uri("file:./mlruns")

experiment_name = "clinical"
existing_experiment = mlflow.get_experiment_by_name(experiment_name)
if existing_experiment is not None:
    experiment_id = existing_experiment.experiment_id
    overwrite_exp = input(f"DO YOU WANT TO OVERWRITE EXISTING {experiment_name} EXPERIMENT? [Y/N]: ")
    if overwrite_exp.lower() == "y":
        mlflow.delete_experiment(experiment_id)
        subprocess.run(["mlflow", "gc", "--experiment-ids", experiment_id], check=True)
    else:
        print("To run the code further, you need to overwrite existing experiment. Please modify code otherwise.")
        exit()

mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)
print(f"Created new experiment for {experiment_name}")

df = pd.read_csv("../datasets/patients_fold.csv")
rechecked_patient_ids = [2209722160, 2209801259, 2209801421, 2209802027, 2209802125]
df = df.loc[df["patient_id"] != 2209801848]
df.loc[df["patient_id"].isin(rechecked_patient_ids), "morphology"] = 0

base_dir = "experiments/clinical"
os.makedirs(f"{base_dir}/train/figures", exist_ok=True)
os.makedirs(f"{base_dir}/test/figures", exist_ok=True)

feature_columns = ['Age', 'Gender', 'Haemoglobin', 'MCV', 'White cell count',
                   'Neutrophil count', 'Monocyte count', 'Platelet count',
                   'Blast percentage (PB)', 'LDH']
target_column = 'morphology'
booster = 'gbtree'

def train_single_sklearn(x, y, seed, feature_names):
    """
    Train a single XGBoost classifier
    """
    params = {
        'booster': booster,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'nthread': 4,
        'seed': seed
    }

    model = xgb.XGBClassifier(**params, importance_type='gain', validate_parameters=True)
    df_x = pd.DataFrame(x, columns=feature_names)
    model.fit(df_x, y)
    return model, params

with mlflow.start_run(run_name="train-val") as parent_run:
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    aurocs = []

    for i in range(5):
        with mlflow.start_run(run_name=f"fold_{i}", nested=True):
            set_col = f'set{i}'
            x_train = df[df[set_col] == 'train'][feature_columns].values
            y_train = df[df[set_col] == 'train'][target_column].values
            x_val = df[df[set_col] == 'val'][feature_columns].values
            y_val = df[df[set_col] == 'val'][target_column].values

            model, params = train_single_sklearn(x_train, y_train, seed=123, feature_names=feature_columns)

            df_x_val = pd.DataFrame(x_val, columns=feature_columns)
            preds = model.predict(df_x_val)
            pred_probs = model.predict_proba(df_x_val)[:, 1]

            accuracy = accuracy_score(y_val, preds)
            precision = precision_score(y_val, preds, zero_division=0)
            recall = recall_score(y_val, preds, zero_division=0)
            f1 = f1_score(y_val, preds, zero_division=0)
            auroc = roc_auc_score(y_val, pred_probs)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("auroc", auroc)

            metrics_path = f"{base_dir}/train/metrics_patient.csv"
            save_metrics_csv(i, accuracy, precision, recall, f1, auroc, metrics_path, train=True)

            roc_path = f"{base_dir}/train/figures/roc_fold_{i}.png"
            plot_save_roc_curve(y_val, pred_probs, roc_path)
            mlflow.log_artifact(roc_path)

            conf_mat = confusion_matrix(y_val, preds)
            conf_mat_path = f"{base_dir}/train/figures/confusion_matrix_fold_{i}.png"
            plot_confusion_matrix(conf_mat, num_classes=2, figure_path=conf_mat_path, title=f"Confusion Matrix Fold {i}")
            mlflow.log_artifact(conf_mat_path)

            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            aurocs.append(auroc)

            print(f"Fold {i} - Accuracy: {round(accuracy*100, 2)}%, Precision: {round(precision*100, 2)}%, "
                  f"Recall: {round(recall*100, 2)}%, F1: {round(f1*100, 2)}%, AUROC: {round(auroc*100, 2)}%")

            model._Booster.__del__()
            del model
            gc.collect()

    mlflow.log_metric("mean_accuracy", np.mean(accuracies))
    mlflow.log_metric("std_accuracy", np.std(accuracies))
    mlflow.log_metric("mean_precision", np.mean(precisions))
    mlflow.log_metric("std_precision", np.std(precisions))
    mlflow.log_metric("mean_recall", np.mean(recalls))
    mlflow.log_metric("std_recall", np.std(recalls))
    mlflow.log_metric("mean_f1", np.mean(f1_scores))
    mlflow.log_metric("std_f1", np.std(f1_scores))
    mlflow.log_metric("mean_auroc", np.mean(aurocs))
    mlflow.log_metric("std_auroc", np.std(aurocs))
    
    mlflow.log_artifact(metrics_path)

    print(f"Mean Accuracy: {round(np.mean(accuracies)*100, 2)}% ± {round(np.std(accuracies)*100, 2)}%")
    print(f"Mean Precision: {round(np.mean(precisions)*100, 2)}% ± {round(np.std(precisions)*100, 2)}%")
    print(f"Mean Recall: {round(np.mean(recalls)*100, 2)}% ± {round(np.std(recalls)*100, 2)}%")
    print(f"Mean F1: {round(np.mean(f1_scores)*100, 2)}% ± {round(np.std(f1_scores)*100, 2)}%")
    print(f"Mean AUROC: {round(np.mean(aurocs)*100, 2)}% ± {round(np.std(aurocs)*100, 2)}%")
    
if mlflow.active_run():
    mlflow.end_run()

with mlflow.start_run(run_name="test"):
    set_col = 'set0'
    train_val_mask = (df[set_col] == 'train') | (df[set_col] == 'val')
    x_train_val = df[train_val_mask][feature_columns].values
    y_train_val = df[train_val_mask][target_column].values
    x_test = df[df[set_col] == 'test'][feature_columns].values
    y_test = df[df[set_col] == 'test'][target_column].values

    model, params = train_single_sklearn(x_train_val, y_train_val, seed=123, feature_names=feature_columns)

    df_x_test = pd.DataFrame(x_test, columns=feature_columns)
    preds = model.predict(df_x_test)
    pred_probs = model.predict_proba(df_x_test)[:, 1]

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auroc = roc_auc_score(y_test, pred_probs)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("auroc", auroc)

    metrics_path = f"{base_dir}/test/metrics.csv"
    save_metrics_csv("Patient", accuracy, precision, recall, f1, auroc, metrics_path, train=False)
    mlflow.log_artifact(metrics_path)

    roc_path = f"{base_dir}/test/figures/roc_final.png"
    plot_save_roc_curve(y_test, pred_probs, roc_path)
    mlflow.log_artifact(roc_path)

    conf_mat = confusion_matrix(y_test, preds)
    conf_mat_path = f"{base_dir}/test/figures/confusion_matrix_final.png"
    plot_confusion_matrix(conf_mat, num_classes=2, figure_path=conf_mat_path, title="Confusion Matrix Final Model")
    mlflow.log_artifact(conf_mat_path)

    print(f"Final Model - Accuracy: {round(accuracy*100, 2)}%, Precision: {round(precision*100, 2)}%, "
          f"Recall: {round(recall*100, 2)}%, F1: {round(f1*100, 2)}%, AUROC: {round(auroc*100, 2)}%")