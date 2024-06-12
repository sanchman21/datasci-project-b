import pandas as pd


df = pd.read_csv('datasets\merged.csv')

results = pd.DataFrame()
sets = ['set0', 'set1', 'set2', 'set3', 'set4']
for set_name in sets:
    total_patients = df['patient_id'].nunique()
    
    train_patients = df[df[set_name] == 'train']['patient_id'].nunique()
    test_patients = df[df[set_name] == 'test']['patient_id'].nunique()
    
    train_morph_0 = df[(df[set_name] == 'train') & (df['morphology'] == 0)]['patient_id'].nunique()
    train_morph_1 = df[(df[set_name] == 'train') & (df['morphology'] == 1)]['patient_id'].nunique()
    test_morph_0 = df[(df[set_name] == 'test') & (df['morphology'] == 0)]['patient_id'].nunique()
    test_morph_1 = df[(df[set_name] == 'test') & (df['morphology'] == 1)]['patient_id'].nunique()

    results = results._append({
        'Set': set_name,
        'Total Patients': total_patients,
        'Train Patients Count': train_patients,
        'Test Patients Count': test_patients,
        'Train Morphology CMML': train_morph_0,
        'Train Morphology normal': train_morph_1,
        'Test Morphology CMML': test_morph_0,
        'Test Morphology normal': test_morph_1
    }, ignore_index=True)

results.columns = ['Set', 'Total Patients', 'Train Patients Count', 'Test Patients Count', 
                   'Train Morphology CMML', 'Train Morphology normal', 'Test Morphology CMML', 'Test Morphology normal']
print(results)
