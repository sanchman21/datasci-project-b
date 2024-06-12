import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = pd.read_csv('nutro-patient-split.csv')

results = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1_score': []
}

for i in range(5):
    train_data = data[data[f'set{i}'] == 'train']
    test_data = data[data[f'set{i}'] == 'test']
    
    X_train = train_data.drop(['morphology', 'image_path','dataset','patient_id', 'patient_id', 'set0', 'set1', 'set2', 'set3', 'set4'], axis=1)
    y_train = train_data['morphology']
    X_test = test_data.drop(['morphology', 'image_path','dataset','patient_id', 'patient_id', 'set0', 'set1', 'set2', 'set3', 'set4'], axis=1)
    y_test = test_data['morphology']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression()
    
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    results['accuracy'].append(accuracy_score(y_test, y_pred))
    results['precision'].append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
    results['recall'].append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
    results['f1_score'].append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

print("Cross-validation results:")
for metric, scores in results.items():
    print(f"{metric}: {sum(scores)/len(scores):.3f} ± {pd.Series(scores).std():.3f}")
