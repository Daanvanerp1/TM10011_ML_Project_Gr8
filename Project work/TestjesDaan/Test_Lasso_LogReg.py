#%%
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
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

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

#Data preprocessing
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

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#Feature selection using logistic regression
lasso_selector = LogisticRegression(
    solver='liblinear',
    l1_ratio=1.0,
    C=0.05,
    class_weight='balanced',
    random_state=42,
    max_iter=2000
)
selector_l1 = SelectFromModel(lasso_selector)
X_train_lasso = selector_l1.fit_transform(X_train_selected, y_train)
X_test_lasso = selector_l1.transform(X_test_selected)

print(f'features before LASSO: {X_train_selected.shape[1]}')
print(f'features after LASSO: {X_train_lasso.shape[1]}')

#Hyperparameter tuning with gridsearch
param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10,100],
    'penalty': ['l2'],
    'solver': ['liblinear', 'lbfgs'],
}

lr_base = LogisticRegression(class_weight='balanced', random_state=42, max_iter=2000)
grid_search_lr = GridSearchCV(
    estimator=lr_base,
    param_grid=param_grid_lr,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search_lr.fit(X_train_lasso, y_train)
best_lr_model = grid_search_lr.best_estimator_

print(f'\nBest Train ROC-AUC(Cross-Validation): {grid_search_lr.best_score_:.4f}')
print(f'Best Hyperparameters: {grid_search_lr.best_params_}')

#Evaluate the best model on the test set
y_test_pred_lr = best_lr_model.predict(X_test_lasso)
y_test_proba_lr = best_lr_model.predict_proba(X_test_lasso)[:, 1]
test_roc_auc_lr = roc_auc_score(y_test, y_test_proba_lr)
test_accuracy_lr = accuracy_score(y_test, y_test_pred_lr)
test_recall_lr = recall_score(y_test, y_test_pred_lr)
test_precision_lr = precision_score(y_test, y_test_pred_lr)
print(f'\nTest ROC-AUC: {test_roc_auc_lr:.4f}')
print(f'Test Accuracy: {test_accuracy_lr:.4f}')
print(f'Test Recall: {test_recall_lr:.4f}')
print(f'Test Precision: {test_precision_lr:.4f}')

# %%
