import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('datasets/patient.csv')

X = data.drop(['morphology', 'dataset', 'patient_id'], axis=1)
y = data['morphology']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted', zero_division=0),
    'recall': make_scorer(recall_score, average='weighted', zero_division=0),
    'f1_score': make_scorer(f1_score, average='weighted', zero_division=0)
}

cv_results = cross_validate(rf_classifier, X_scaled, y, cv=cv, scoring=scoring, return_train_score=True)

print("Cross-validation results:")
for metric, scores in cv_results.items():
    print(f"{metric}: {scores.mean():.3f} ± {scores.std():.3f}")
