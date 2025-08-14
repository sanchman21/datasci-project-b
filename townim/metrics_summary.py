import pandas as pd
import os
import mlflow
import subprocess

data_types = ["neutrophil", "monocyte", "neutrophil_clinical", "monocyte_clinical", "monocyte_new_normals", "clinical", "segmented_monocyte", "segmented_monocyte_new_normals"]
metrics = ["accuracy", "precision", "recall", "f1", "auroc"]
levels = ["image", "image_tta", "patient"]
exp_dir = "./experiments"
summary_dir = exp_dir + "/summary"
os.makedirs(summary_dir, exist_ok=True)
experiment_name = "summary"
existing_experiment = mlflow.get_experiment_by_name(experiment_name)

if existing_experiment is not None:
    experiment_id = existing_experiment.experiment_id
    overwrite_exp = input(f"DO YOU WANT TO OVERWRITE EXISTING {experiment_name}? [Y/N]")
    if overwrite_exp.lower() == "y":
        mlflow.delete_experiment(experiment_id)
        subprocess.run(["mlflow", "gc", "--experiment-ids", experiment_id], check=True)
    else:
        print("To run further, overwrite the existing experiment or modify the code.")
        exit()
        
mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)
print(f"Created new MLflow experiment: {experiment_name}")

def log_summary_csv(run_name, data_dict, output_csv):
    with mlflow.start_run(run_name=run_name, nested=True):
        df = pd.DataFrame(data_dict)
        df.to_csv(output_csv, index=False)
        mlflow.log_artifact(output_csv)
        print(f"Saved and logged {output_csv}")
        
with mlflow.start_run(run_name="val-metrics"):
    for level in levels:
        val_data = {"data_type": []}
        for metric in metrics:
            val_data[f"{metric}_mean"] = []
            val_data[f"{metric}_std"] = []
        relevant_data_types = (
            data_types if level == "patient"
            else [dt for dt in data_types if dt not in ["neutrophil_clinical", "monocyte_clinical", "clinical"]]
        )
        
        for data_type in relevant_data_types:
            path = f"{exp_dir}/{data_type}/train/metrics_{level}.csv"
            if os.path.exists(path):
                print(f"Processing {data_type} - {level} (val)")
                df = pd.read_csv(path)
                val_data["data_type"].append(data_type)
                
                for metric in metrics:
                    if metric in df.columns:
                        mean_val = round(df[metric].mean() * 100, 2)
                        std_val = round(df[metric].std() * 100, 2)
                        val_data[f"{metric}_mean"].append(mean_val)
                        val_data[f"{metric}_std"].append(std_val)
                    else:
                        val_data[f"{metric}_mean"].append(None)
                        val_data[f"{metric}_std"].append(None)
                        
        if val_data["data_type"]:
            output_csv = f"{summary_dir}/val_metrics_{level}.csv"
            log_summary_csv(level, val_data, output_csv)
            
if mlflow.active_run():
    mlflow.end_run()
    
with mlflow.start_run(run_name="test-metrics"):
    for level in levels:
        test_data = {"data_type": [], **{metric: [] for metric in metrics}}
        relevant_data_types = (
            data_types if level == "patient"
            else [dt for dt in data_types if dt not in ["neutrophil_clinical", "monocyte_clinical", "clinical"]]
        )
        for data_type in relevant_data_types:
            path = f"{exp_dir}/{data_type}/test/metrics.csv"
            if os.path.exists(path):
                print(f"Processing {data_type} - {level} (test)")
                df = pd.read_csv(path)
                level_map = {"image": "Image W/O TTA", "image_tta": "Image TTA", "patient": "Patient"}
                row = df[df["test type"] == level_map[level]]
                if not row.empty:
                    test_data["data_type"].append(data_type)
                    for metric in metrics:
                        if metric in row.columns:
                            test_data[metric].append(round(row[metric].iloc[0] * 100, 2))
                        else:
                            test_data[metric].append(None)
                            
        if test_data["data_type"]:
            print(test_data)
            output_csv = f"{summary_dir}/test_metrics_{level}.csv"
            log_summary_csv(level, test_data, output_csv)
            
print("Summaries completed and logged to MLflow.")