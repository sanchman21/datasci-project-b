import pandas as pd

def compare_patient_folds(csv_file1, csv_file2):
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)

    mismatches = []

    for fold in range(5):
        set_col = f'set{fold}'
        
        df1_train_patients = set(df1[df1[set_col] == 'train']['patient_id'])
        df1_test_patients = set(df1[df1[set_col] == 'test']['patient_id'])
        
        df2_train_patients = set(df2[df2[set_col] == 'train']['patient_id'])
        df2_test_patients = set(df2[df2[set_col] == 'test']['patient_id'])
        
        if df1_train_patients != df2_train_patients:
            mismatches.append((fold, 'train', df1_train_patients, df2_train_patients))
        
        if df1_test_patients != df2_test_patients:
            mismatches.append((fold, 'test', df1_test_patients, df2_test_patients))

    if mismatches:
        print("Patient set mismatch detected in the following folds:")
        for fold, set_type, patients1, patients2 in mismatches:
            print(f"Fold {fold} - {set_type} set mismatch:")
            print(f"File 1 {set_type} patients: {patients1}")
            print(f"File 2 {set_type} patients: {patients2}")
    else:
        print("No mismatch detected. All folds and sets match perfectly.")

if __name__ == "__main__":
    csv_file1 = 'datasets/neutrophil.csv'
    csv_file2 = 'datasets\monocyte_reassigned.csv'  
    
    compare_patient_folds(csv_file1, csv_file2)
