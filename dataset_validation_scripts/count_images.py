import pandas as pd


df = pd.read_csv('new_split2.csv')

results = pd.DataFrame()

sets = ['set0', 'set1', 'set2', 'set3', 'set4']
for set_name in sets:
    total_patients = df.groupby(set_name)['patient_id'].nunique()
    
    train_test_distribution = df.groupby([set_name, 'patient_id']).size().unstack(fill_value=0).sum(axis=1).reset_index()
    train_count = train_test_distribution[train_test_distribution[set_name] == 'train'][0].sum()
    test_count = train_test_distribution[train_test_distribution[set_name] == 'test'][0].sum()

    train_cmml_count = df[(df[set_name] == 'train') & (df['morphology'] == 0)].shape[0]
    train_normal_count = df[(df[set_name] == 'train') & (df['morphology'] == 1)].shape[0]
    test_cmml_count = df[(df[set_name] == 'test') & (df['morphology'] == 0)].shape[0]
    test_normal_count = df[(df[set_name] == 'test') & (df['morphology'] == 1)].shape[0]

    results = results._append({
        'Set': set_name,
        'Total Patients': total_patients.sum(),
        'Train Data Count': train_count,
        'Test Data Count': test_count,
        'Train CMML': train_cmml_count,
        'Train Normal': train_normal_count,
        'Test CMML': test_cmml_count,
        'Test Normal': test_normal_count
    }, ignore_index=True)

results.columns = ['Set', 'Total Patients', 'Train Data Count', 'Test Data Count', 
                   'Train CMML', 'Train Normal', 'Test CMML', 'Test Normal']
print(results)
