#Package imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, recall_score, precision_score
from worcgist.load_data import load_data

#Load data
df = pd.DataFrame(load_data())
print(f'The number of samples: {len(df.index)}')
print(f'The number of columns: {len(df.columns)}')
print(df.head())

#Data inspection
print(f'\nTotal missing values: {df.isnull().sum().sum()}')                         #Missing data
print(f'\nLabel distribution:')
print(df['label'].value_counts())
plt.bar(df['label'].value_counts().index, df['label'].value_counts().values)        #Label distribution plot
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Label Distribution')
#plt.show()

df.info()                                                                          #Data info

#Data preprocessing                                                      #Drop ID column
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])                             #Label encoding

X = df.drop(['label'], axis=1)                                                        
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f'Training set shape: {X_train.shape}')                                       #Training set shape
print(f'Testing set shape: {X_test.shape}')                                         #Testing set shape
print(f'Training set label distribution:\n{y_train.value_counts()}')                 #Training set label distribution
print(f'Testing set label distribution:\n{y_test.value_counts()}')                   #Testing set label distribution

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)                                              #Feature scaling
X_test_scaled = scaler.transform(X_test)                                                     #Feature scaling

X_train_winsorized = np.clip(X_train_scaled, np.percentile(X_train_scaled, 1, axis=0), np.percentile(X_train_scaled, 99, axis=0))   #Outlier handling
X_test_winsorized = np.clip(X_test_scaled, np.percentile(X_train_scaled, 1, axis=0), np.percentile(X_train_scaled, 99, axis=0))     #Outlier handling


#Baseline cross-validation
Classifiers = {'logistic_regression': LogisticRegression(class_weight='balanced',max_iter=10000, random_state=42),
               'random_forest': RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
               'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# for name, clf in Classifiers.items():
#     cv_scores = cross_val_score(clf, X_train_winsorized, y_train, cv=cv, scoring='roc_auc')
#     mean_score = cv_scores.mean()
#     std_score = cv_scores.std()
#     print(f'{name}:')
#     print(f'Scores per fold: {np.round(cv_scores, 4)}')
#     print(f'Mean ROC-AUC: {mean_score:.4f} ± {std_score:.4f}\n')


#Applying variance threshold and correlation filtering
var_threshold = VarianceThreshold(threshold=0.01)
X_train_var = var_threshold.fit_transform(X_train_winsorized)
X_test_var = var_threshold.transform(X_test_winsorized)
print(f'Features after variance thresholding: {X_train_var.shape[1]}')

corr_matrix = np.abs(np.corrcoef(X_train_var, rowvar=False))
upper_tri = np.triu(corr_matrix, k=1)
threshold = 0.90
to_drop = [column for column in range(upper_tri.shape[1]) if any(upper_tri[:, column] > threshold)]
X_train_selected = np.delete(X_train_var, to_drop, axis=1)
X_test_selected = np.delete(X_test_var, to_drop, axis=1)
print(f'Features after correlation filtering: {X_train_selected.shape[1]}')
print(f'Total features removed: {X_train_winsorized.shape[1] - X_train_selected.shape[1]}')

scoring = ['roc_auc', 'accuracy', 'precision', 'recall']

# for name, clf in Classifiers.items():
#     cv_results = cross_validate(clf, X_train_selected, y_train, cv=cv, scoring=scoring)
#     print(f'{name}:')
#     print(f"  Mean ROC-AUC:   {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
#     print(f"  Mean Accuracy:  {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
#     print(f"  Mean Recall:    {cv_results['test_recall'].mean():.4f} ± {cv_results['test_recall'].std():.4f}")
#     print(f"  Mean Precision: {cv_results['test_precision'].mean():.4f} ± {cv_results['test_precision'].std():.4f}\n")


#Model based feature selection
# Random forest selector
rf_selector = RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=42)
selector_model = SelectFromModel(rf_selector, threshold=-np.inf, max_features=30)

X_train_final = selector_model.fit_transform(X_train_selected, y_train)
X_test_final = selector_model.transform(X_test_selected)
print(f'Features after model-based selection: {X_train_final.shape[1]}')

# for name, clf in Classifiers.items():
#     cv_results = cross_validate(clf, X_train_final, y_train, cv=cv, scoring=scoring)
#     print(f'{name}:')
#     print(f"  Mean ROC-AUC:   {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
#     print(f"  Mean Accuracy:  {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
#     print(f"  Mean Recall:    {cv_results['test_recall'].mean():.4f} ± {cv_results['test_recall'].std():.4f}")
#     print(f"  Mean Precision: {cv_results['test_precision'].mean():.4f} ± {cv_results['test_precision'].std():.4f}\n")

#Grid search for best hyperparameters
parameter_grid = {'n_estimators': [100, 200,300,],
                  'max_depth': [None, 5, 10,],
                  'min_samples_split': [2, 5, 10],
                  'min_samples_leaf': [1, 2, 4]}
rf_base = RandomForestClassifier(class_weight='balanced', random_state=42)
grid_search = GridSearchCV(estimator=rf_base, param_grid=parameter_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)
grid_search.fit(X_train_final, y_train)

print(f'Best ROC-AUC: {grid_search.best_score_:.4f}')
print(f'Best hyperparameters:')
for param, value in grid_search.best_params_.items():
    print(f'  {param}: {value}')

best_rf = grid_search.best_estimator_

#Evaluation on test set
y_test_pred = best_rf.predict(X_test_final)
y_test_pred_proba = best_rf.predict_proba(X_test_final)[:, 1]
test_roc_auc = roc_auc_score(y_test, y_test_pred_proba)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
print(f'\nTest ROC-AUC: {test_roc_auc:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Test Recall: {test_recall:.4f}')
print(f'Test Precision: {test_precision:.4f}\n')
