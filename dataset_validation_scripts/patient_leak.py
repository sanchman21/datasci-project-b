import pandas as pd

def check_patient_leak(csv_file):
    df = pd.read_csv(csv_file)
    leaks = []

    for fold in range(5):
        set_col = f'set{fold}'
        fold_df = df[df[set_col].isin(['train', 'test'])]
        patient_groups = fold_df.groupby('patient_id')[set_col].nunique()

        leaked_patients = patient_groups[patient_groups > 1].index.tolist()
        if leaked_patients:
            leaks.append((fold, leaked_patients))

    if leaks:
        print("Patient leak detected in the following folds:")
        for fold, patients in leaks:
            print(f"Fold {fold}: Leaked Patients {patients}")
    else:
        print("No patient leak detected across all folds.")

if __name__ == "__main__":
    csv_file = 'datasets\\reassigned_monocyte.csv'
    check_patient_leak(csv_file)
