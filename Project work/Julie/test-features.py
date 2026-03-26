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
X_train_selected = selector.fit_transform(X_train_winsorized)
X_test_selected = selector.transform(X_test_winsorized)

print(f"Features remaining after variance thresholding: {X_train_selected.shape[1]}")

# %%
# feature selection with Mann Whitney U test
from scipy.stats import mannwhitneyu

def mann_whitney_u_test(X, y):
    selected_features_mwu = []
    for i in range(X.shape[1]):
        group0 = X[y == 0, i]
        group1 = X[y == 1, i]
        stat, p_value = mannwhitneyu(group0, group1)
        if p_value < 0.05:
            selected_features_mwu.append(i)
    return selected_features_mwu

# Run the test on training data

selected_indices_mwu = mann_whitney_u_test(X_train_selected, y_train)

# Filter the training and testing sets to only keep these features
X_train_final_mwu = X_train_selected[:, selected_indices_mwu]
X_test_final_mwu = X_test_selected[:, selected_indices_mwu]

print(f"Features remaining after Mann-Whitney U test: {X_train_final_mwu.shape[1]}")

# %%
# feature selection with spearman correlation
from scipy.stats import spearmanr
def spearman_correlation(X, y):
    selected_features_pc = []
    for i in range(X.shape[1]):
        corr, p_value = spearmanr(X[:, i], y)
        if p_value < 0.05:
            selected_features_pc.append(i)
    return selected_features_pc

# Run the test on training data
selected_indices_pc = spearman_correlation(X_train_selected, y_train)   

# Filter the training and testing sets to only keep these features
X_train_final_pc = X_train_selected[:, selected_indices_pc]
X_test_final_pc = X_test_selected[:, selected_indices_pc]

print(f"Features remaining after Spearman correlation: {X_train_final_pc.shape[1]}")

# %%
# feature selection with maximum relevance minimum redundancy (mRMR)
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
def mrmr(X, y, k):
    # 1. Relevance: Correlation between each feature and the label y
    relevance = mutual_info_classif(X, y)
    
    selected_features = []
    # Start with the most relevant feature
    selected_features.append(np.argmax(relevance))
    
    for _ in range(1, k):
        mrmr_scores = []
        for i in range(X.shape[1]):
            if i in selected_features:
                mrmr_scores.append(-np.inf) # Skip already selected
                continue
            
            # 2. Redundancy: Correlation with already selected features
            # Note: This is computationally expensive in a manual loop!
            redundancy = 0
            for selected in selected_features:
                feat_i = X[:, i].reshape(-1, 1)
                feat_selected = X[:, selected] # This is treated as the 'target'
                
                # USE REGRESSION HERE because feat_selected is continuous!
                redundancy += mutual_info_regression(feat_i, feat_selected)[0]
            
            redundancy /= len(selected_features)
            mrmr_scores.append(relevance[i] - redundancy)
            
        selected_features.append(np.argmax(mrmr_scores))
        
    return selected_features

# Run the test on training data
selected_indices_mrmr = mrmr(X_train_selected, y_train, k=20)  # Select top 20 features

# Filter the training and testing sets to only keep these features
X_train_final_mrmr = X_train_selected[:, selected_indices_mrmr]
X_test_final_mrmr = X_test_selected[:, selected_indices_mrmr]

print(f"Features remaining after mRMR: {X_train_final_mrmr.shape[1]}")

# %%
# Feature selection with ANOVA F-test
from sklearn.feature_selection import f_classif
def anova_f_test(X, y):
    f_values, p_values = f_classif(X, y)
    selected_features_anova = [i for i in range(len(p_values)) if p_values[i] < 0.05]
    return selected_features_anova

# Run the test on training data
selected_indices_anova = anova_f_test(X_train_selected, y_train)

# Filter the training and testing sets to only keep these features
X_train_final_anova = X_train_selected[:, selected_indices_anova]
X_test_final_anova = X_test_selected[:, selected_indices_anova]

# %%
# Feature selection with mutual information
from sklearn.feature_selection import mutual_info_classif
def mutual_information(X, y):
    mi_scores = mutual_info_classif(X, y)
    selected_features_mi = [i for i in range(len(mi_scores)) if mi_scores[i] > 0.01]  # Threshold can be adjusted
    return selected_features_mi

# Run the test on training data
selected_indices_mi = mutual_information(X_train_selected, y_train)

# Filter the training and testing sets to only keep these features
X_train_final_mi = X_train_selected[:, selected_indices_mi]
X_test_final_mi = X_test_selected[:, selected_indices_mi]

print(f"Features remaining after mutual information: {X_train_final_mi.shape[1]}")

# %%
# Feature selection with Lasso regression
from sklearn.linear_model import Lasso
def lasso_feature_selection(X, y, alpha=0.01):
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)
    selected_features_lasso = [i for i in range(len(lasso.coef_)) if lasso.coef_[i] != 0]
    return selected_features_lasso

# Run the test on training data
selected_indices_lasso = lasso_feature_selection(X_train_selected, y_train)

# Filter the training and testing sets to only keep these features
X_train_final_lasso = X_train_selected[:, selected_indices_lasso]
X_test_final_lasso = X_test_selected[:, selected_indices_lasso]

print(f"Features remaining after Lasso regression: {X_train_final_lasso.shape[1]}")

# %%
# Feature selection with Recursive Feature Elimination (RFE)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
def rfe_feature_selection(X, y, n_features_to_select=20):
    model = LogisticRegression(max_iter=1000)
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
    rfe.fit(X, y)
    selected_features_rfe = [i for i in range(len(rfe.support_)) if rfe.support_[i]]
    return selected_features_rfe

# Run the test on training data
selected_indices_rfe = rfe_feature_selection(X_train_selected, y_train, n_features_to_select=20)

# Filter the training and testing sets to only keep these features
X_train_final_rfe = X_train_selected[:, selected_indices_rfe]
X_test_final_rfe = X_test_selected[:, selected_indices_rfe]

print(f"Features remaining after RFE: {X_train_final_rfe.shape[1]}")

# %%
# Feature selection with forward-backward selection
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
def forward_backward_selection(X, y, n_features_to_select=20):
    model = LogisticRegression(max_iter=1000)
    sfs = SequentialFeatureSelector(model, n_features_to_select=n_features_to_select, direction='forward')
    sfs.fit(X, y)
    selected_features_forward = [i for i in range(len(sfs.get_support())) if sfs.get_support()[i]]
    
    sfs_backward = SequentialFeatureSelector(model, n_features_to_select=n_features_to_select, direction='backward')
    sfs_backward.fit(X, y)
    selected_features_backward = [i for i in range(len(sfs_backward.get_support())) if sfs_backward.get_support()[i]]
    
    # Combine forward and backward selected features
    selected_features_combined = list(set(selected_features_forward) | set(selected_features_backward))
    
    return selected_features_combined

# Run the test on training data
selected_indices_forward_backward = forward_backward_selection(X_train_selected, y_train, n_features_to_select=20)

# Filter the training and testing sets to only keep these features
X_train_final_forward_backward = X_train_selected[:, selected_indices_forward_backward]
X_test_final_forward_backward = X_test_selected[:, selected_indices_forward_backward]

print(f"Features remaining after forward-backward selection: {X_train_final_forward_backward.shape[1]}")    
# %%
