# ## Data loading and cleaning
# Below are functions to load the dataset of your choice. After that, it is all up to you to create and evaluate a classification method. Beware, there may be missing values in these datasets. Good luck!
#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from worcgist.load_data import load_data
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from scipy.stats.mstats import winsorize
from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import StratifiedKFold
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectFromModel
from scipy.stats import mannwhitneyu
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix


#%% Data loading functions. Uncomment the one you want to use

data = load_data()
print(f'The number of samples: {len(data.index)}')

print(f'The number of columns: {len(data.columns)}')

# %% Data inspection and cleaning
# show data head pandas dataframe
print(f'The first 5 rows of the dataset:')
print(data.head())

#Show missing data
total_missing = data.isnull().sum().sum()
print(f'\nTotal missing values: {total_missing}')

#Show label distribution
print(f'\nLabel distribution:')
print(data['label'].value_counts())
plt.bar(data['label'].value_counts().index, data['label'].value_counts().values)
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Label Distribution')
plt.show()

#Show data info
print("\n--- Data Info ---")
data.info()

#Show data statistics
print(data.describe())

#Show data correlation
features_subset = data.select_dtypes(include=['float64', 'int64']).iloc[:, :15]
corr_matrix = features_subset.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title('Correlatie Heatmap (eerste 15 features)')
plt.show()

# %%
#Show outliers
# Maak histogrammen van de eerste 4 features
data.iloc[:, 1:5].hist(figsize=(10, 8), bins=30, color='skyblue', edgecolor='black')
plt.suptitle('Histogrammen van geselecteerde features', y=1.02)
plt.tight_layout()
plt.show()

# %% Data Preprocessing
# Label encoding (Switch Gist and non-Gist to 1 and 0 respectively) and delete first column (ID)
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# Split data into training and testing sets, stratify and use 20% of the data for testing
X = data.drop(['label'], axis=1)
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# show the shape and distribution of the training and testing sets
print(f'Training set shape: {X_train.shape}')
print(f'Testing set shape: {X_test.shape}')
print(f'Training set label distribution: {y_train.value_counts()}')
print(f'Testing set label distribution: {y_test.value_counts()}')

# %%
# Data scaling (Standardization)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Winsorization
X_train_winsorized = winsorize(X_train_scaled, limits=[0.1, 0.1], axis=0)
X_test_winsorized = winsorize(X_test_scaled, limits=[0.1, 0.1], axis=0)

# variance thresholding
selector = VarianceThreshold(threshold=0.1)
X_train_var = selector.fit_transform(X_train_winsorized)
X_test_var = selector.transform(X_test_winsorized)

# correlation filtering
corr_matrix = np.abs(np.corrcoef(X_train_var, rowvar=False))
upper_tri = np.triu(corr_matrix, k=1)
threshold = 0.90
to_drop = [column for column in range(upper_tri.shape[1]) if any(upper_tri[:, column] > threshold)]
X_train_selected = np.delete(X_train_var, to_drop, axis=1)
X_test_selected = np.delete(X_test_var, to_drop, axis=1)
print(f'Features after correlation filtering: {X_train_selected.shape[1]}')
print(f'Total features removed: {X_train_winsorized.shape[1] - X_train_selected.shape[1]}')  
      


# %%

# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# methods = ['Mann-Whitney', 'LASSO', 'Mutual Info']
# cv_results = {m: [] for m in methods}

# for train_idx, val_idx in cv.split(X_train_selected, y_train):
#     # Splitsen in interne train- en validatie-folds
#     X_tr, X_val = X_train_selected[train_idx], X_train_selected[val_idx]
#     y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
#     # 1. TEST: Mann-Whitney U selectie
#     selected_mw = []
#     for i in range(X_tr.shape[1]):
#         _, p = mannwhitneyu(X_tr[y_tr==0, i], X_tr[y_tr==1, i])
#         if p < 0.05: selected_mw.append(i)
    
#     if selected_mw:
#         lr = LogisticRegression(max_iter=1000, class_weight='balanced')
#         lr.fit(X_tr[:, selected_mw], y_tr)
#         probs = lr.predict_proba(X_val[:, selected_mw])[:, 1]
#         cv_results['Mann-Whitney'].append(roc_auc_score(y_val, probs))

# # Print resultaten
# for method, scores in cv_results.items():
#     print(f"{method}: Mean AUC = {np.mean(scores):.3f}")





# %%


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

methods = ['Mann-Whitney', "Mutual-Info", 'LASSO']
cv_results = {m: [] for m in methods}


# 1. Definieer de zoekruimte voor de hyperparameters
param_dist = {
    'C': loguniform(1e-4, 1e2),  # Zoekt breed tussen 0.0001 en 100
    'penalty': ['l1', 'l2'],    # Test zowel L1 (Lasso) als L2 (Ridge)
}

# De basis estimator die we overal gebruiken
lr_base = LogisticRegression(
    max_iter=1000, 
    random_state=42, 
    class_weight="balanced", 
    solver='liblinear'
)



# (De rest van je loop blijft gelijk...)
# for train_idx, val_idx in cv.split(X_train_selected, y_train):
for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train_selected, y_train)):
    # Splitsen in interne train- en validatie-folds
    X_tr, X_val = X_train_selected[train_idx], X_train_selected[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # 1. TEST: Mann-Whitney U selectie
    # We tunen hier de p-waarde drempel
    best_mw_auc = -1
    for p_thresh in [0.001, 0.01, 0.05, 0.1]:
        selected_mw = []
        for i in range(X_tr.shape[1]):
            # Test per feature of er verschil is tussen de klassen
            _, p = mannwhitneyu(X_tr[y_tr==0, i], X_tr[y_tr==1, i])
            if p < p_thresh: selected_mw.append(i)    # ook nog een hyperparameter die kan worden aangepast?
    
    if selected_mw:
        # 2. Initialiseer de Random Search
        random_search = RandomizedSearchCV(
            estimator = lr_base,
            param_distributions=param_dist,
            n_iter=20,           # Probeer 20 willekeurige combinaties
            cv=5,                # Interne cross-validatie (3 folds)
            scoring='roc_auc',
            random_state=42,
            n_jobs=-1            # Gebruik alle processoren voor snelheid
        )
        # 3. Fit de search op de huidige trainings-fold
        random_search.fit(X_tr[:, selected_mw], y_tr)


        if random_search.best_score_ > best_mw_auc:
                best_mw_auc = random_search.best_score_
                best_mw_model = random_search.best_estimator_
                best_mw_idx = selected_mw
        
        
        
        # 4. Pak het beste model en voorspel op de validatie-fold
        best_lr = random_search.best_estimator_
        probs = best_lr.predict_proba(X_val[:, selected_mw])[:, 1]
     
        cv_results['Mann-Whitney'].append(roc_auc_score(y_val, probs))

    
    #2. TEST: MUTUAL INFORMATION ---
    # We tunen hier het aantal features (k)
    best_mi_auc = -1
    # We testen verschillende groottes voor de feature set
    for k_feat in [5, 10, 20, 50]:
        # Voorkom dat k groter is dan het aantal beschikbare features
        k_actual = min(k_feat, X_tr.shape[1])
        mi_selector = SelectKBest(score_func=mutual_info_classif, k=k_actual)
        mi_selector.fit(X_tr, y_tr)
        selected_mi = mi_selector.get_support(indices=True)

        # Initialiseer de Random Search
        random_search = RandomizedSearchCV(
            estimator = lr_base,
            param_distributions=param_dist,
            n_iter=20,           # Probeer 20 willekeurige combinaties
            cv=5,                # Interne cross-validatie (3 folds)
            scoring='roc_auc',
            random_state=42,
            n_jobs=-1            # Gebruik alle processoren voor snelheid
        )
        random_search.fit(X_tr[:, selected_mi], y_tr)
        
        if random_search.best_score_ > best_mi_auc:
            best_mi_auc = random_search.best_score_
            best_mi_model = random_search.best_estimator_
            best_mi_idx = selected_mi

    auc_mi = roc_auc_score(y_val, best_mi_model.predict_proba(X_val[:, best_mi_idx])[:, 1])
    cv_results['Mutual-Info'].append(auc_mi)


        # # 2. TEST: LASSO SELECTION ---
        # # Feature selection stap via Lasso (L1)
        # lasso_selector = SelectFromModel(
        #     estimator=LogisticRegression(
        #         penalty='l1', 
        #         C=0.1, 
        #         solver='liblinear', 
        #         random_state=42, 
        #         class_weight="balanced"))
        # lasso_selector.fit(X_tr, y_tr)
        # selected_lasso = lasso_selector.get_support(indices=True)
    
        # if len(selected_lasso) > 0:
        #     # Dezelfde hyperparameter search, maar dan op Lasso-features
        #     mu_lasso = RandomizedSearchCV(
        #         estimator=lr_base,
        #         param_distributions=param_dist,
        #         n_iter=20, cv=3, scoring='roc_auc', 
        #         random_state=, n_jobs=-1
        #     )
        #     rs_lasso.fit(X_tr[:, selected_lasso], y_tr)
        
        #     # Testen op validatie fold
        #     auc_lasso = roc_auc_score(y_val, rs_lasso.best_estimator_.predict_proba(X_val[:, selected_lasso])[:, 1])
        #     cv_results['Lasso-Selection'].append(auc_lasso)

    print(f"Fold {fold_idx + 1} gereed.")


# --- 3. RESULTATEN ---
for m, scores in cv_results.items():
    print(f"{m:15} | AUC: {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")


# %%


# Print resultaten
for method, scores in cv_results.items():
    print(f"{method}: Mean AUC = {np.mean(scores):.3f}")




# %%



