import pandas as pd 

root = "./experiments"
with_TTA = True
is_neutrophil = True
data_type = "neutrophil" if is_neutrophil else "monocyte"
results = pd.DataFrame(columns=["Fold", "Loss", "Accuracy", "Precision", "Recall", "F1", "AUROC"])

for set_id in range(5):
    if with_TTA:
        metrics_path = root + f"/{data_type}_fold_{set_id}_with_TTA/train_time_metrics.csv"
    else:
        metrics_path = root + f"/{data_type}_fold_{set_id}/train_time_metrics.csv"
    df = pd.read_csv(metrics_path)
    new_df = pd.Series({
        "Fold": set_id,
        "Loss": round(df["Val Loss"].iloc[-1], 3),
        "Accuracy": round(df["Val Accuracy"].iloc[-1], 3),
        "Precision": round(df["Val Precision"].iloc[-1], 3),
        "Recall": round(df["Val Recall"].iloc[-1], 3),
        "F1": round(df["Val F1"].iloc[-1], 3),
        "AUROC": round(df["Val AUROC"].iloc[-1], 3),
    })
    results = pd.concat([results, new_df.to_frame().T], ignore_index=True)

results.to_csv(root + "/all_train_time_metrics.csv", index=False)