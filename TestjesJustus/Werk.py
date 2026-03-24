# ## GIST Feature Selection and Classification
# This script performs feature selection and classification on GIST data using 
# a methodologically sound pipeline that prevents data leakage.

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
import warnings
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.exceptions import ConvergenceWarning

# Onderdruk convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# %% 1. Data Loading
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'GIST_features_filtered.csv')
data = pd.read_csv(file_path)

# Data opschonen: Weg met ID en zorg dat de rest numeriek is (behalve de label)
if 'ID' in data.columns:
    data = data.drop(columns=['ID'])

# Behoud label (ook als het tekst is) en filter de rest op numeriek
cols_to_keep = ['label'] + list(data.drop(columns=['label']).select_dtypes(include=[np.number]).columns)
data = data[cols_to_keep]

print(f'Total samples: {len(data.index)}')
print(f'Total features (before cleaning): {len(data.columns) - 1}')

# %% 2. Data Inspection
print(f'\nFirst 5 rows:')
print(data.head())

print(f'\nLabel distribution:')
print(data['label'].value_counts())

# Heatmap (eerste 15 features)
plt.figure(figsize=(10, 8))
sns.heatmap(data.iloc[:, 1:16].corr(), cmap='coolwarm', annot=False)
plt.title('Correlatie Heatmap (eerste 15 features)')
plt.show()

# %% 3. Preprocessing
# Label encoding
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# Scheid Features (X) en Target (y)
X = data.drop(columns=['label'])
y = data['label']

# Verwijder constante features
X = X.loc[:, X.std() > 0]
print(f'Features remaining after removing constants: {X.shape[1]}')

# Split data (20% testset)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# %% 4. Feature Selection Tuning (Leakage-Free)
k_values = [10, 20, 30, 40, 50]
methods = ["Mann-Whitney U", "Spearman Correlation"]
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

def evaluate_method(method_name, k, X_data, y_data):
    """
    Voert cross-validatie uit waarbij scaling en feature selection BINNEN de folds gebeuren.
    """
    metrics = {'auc': [], 'f1': []}
    
    for train_idx, val_idx in skf.split(X_data, y_data):
        X_f_train, X_f_val = X_data.iloc[train_idx].copy(), X_data.iloc[val_idx].copy()
        y_f_train, y_f_val = y_data.iloc[train_idx].copy(), y_data.iloc[val_idx].copy()
        
        # Scaling
        scaler = RobustScaler()
        X_f_train_scaled = scaler.fit_transform(X_f_train)
        X_f_val_scaled = scaler.transform(X_f_val)
        
        # Selection
        if method_name == "Mann-Whitney U":
            p_values = [mannwhitneyu(X_f_train_scaled[y_f_train == 0, i], 
                                     X_f_train_scaled[y_f_train == 1, i])[1] 
                        for i in range(X_f_train_scaled.shape[1])]
            selected_indices = np.argsort(p_values)[:k]
        else:
            corrs = [abs(spearmanr(X_f_train_scaled[:, i], y_f_train)[0]) 
                     for i in range(X_f_train_scaled.shape[1])]
            selected_indices = np.argsort(corrs)[-k:]
        
        # Training
        lr = LogisticRegression(max_iter=10000)
        lr.fit(X_f_train_scaled[:, selected_indices], y_f_train)
        
        # Testing
        y_prob = lr.predict_proba(X_f_val_scaled[:, selected_indices])[:, 1]
        y_pred = lr.predict(X_f_val_scaled[:, selected_indices])
        
        metrics['auc'].append(roc_auc_score(y_f_val, y_prob))
        metrics['f1'].append(f1_score(y_f_val, y_pred))
        
    return np.mean(metrics['auc']), np.mean(metrics['f1'])

all_results = []
print("\nStarten met parameter tuning...")
for method in methods:
    for k in k_values:
        mean_auc, mean_f1 = evaluate_method(method, k, X_train, y_train)
        all_results.append({'Method': method, 'k': k, 'Mean_AUC': mean_auc, 'Mean_F1': mean_f1})
        print(f"Gereed: {method} (k={k}) -> AUC: {mean_auc:.4f}")

# Overzicht tonen
results_df = pd.DataFrame(all_results)
print("\n--- Cross-Validatie Resultaten ---")
print(results_df.sort_values(by='Mean_AUC', ascending=False))

# %% 5. Evaluatie op Testset
best_row = results_df.loc[results_df['Mean_AUC'].idxmax()]
best_method, best_k = best_row['Method'], best_row['k']

print(f"\nBeste model: {best_method} (k={best_k})")

# Finaal model trainen
scaler_final = RobustScaler()
X_tr_sc = scaler_final.fit_transform(X_train)
X_te_sc = scaler_final.transform(X_test)

if best_method == "Mann-Whitney U":
    p_vals = [mannwhitneyu(X_tr_sc[y_train == 0, i], X_tr_sc[y_train == 1, i])[1] for i in range(X_tr_sc.shape[1])]
    final_indices = np.argsort(p_vals)[:best_k]
else:
    corrs = [abs(spearmanr(X_tr_sc[:, i], y_train)[0]) for i in range(X_tr_sc.shape[1])]
    final_indices = np.argsort(corrs)[-best_k:]

lr_final = LogisticRegression(max_iter=10000)
lr_final.fit(X_tr_sc[:, final_indices], y_train)

# Finale resultaten
y_test_prob = lr_final.predict_proba(X_te_sc[:, final_indices])[:, 1]
y_test_pred = lr_final.predict(X_te_sc[:, final_indices])

print("\n--- TESTSET RESULTATEN ---")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, y_test_prob):.4f}")
print(f"F1-score: {f1_score(y_test, y_test_pred):.4f}")
