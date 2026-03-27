#%% Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import scipy.stats as stats
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, RFE
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, auc, roc_curve, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import mannwhitneyu, loguniform, randint
from xgboost import XGBClassifier
from tqdm.auto import tqdm
from worcgist.load_data import load_data
warnings.filterwarnings('ignore')



#%% Part 1: Data loading and inspection

# Part 1.1: Data loading functions
data = load_data()
print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')


# Part 1.2: Data Inspection
# Show data head
print('\nThe first 5 rows of the dataset:')
print(data.head())

# Show missing data
total_missing = data.isnull().sum().sum()
print(f'\nTotal missing values: {total_missing}')

# Show data info
print('\nData information:')
print(data.info())

# Show label distribution
print('\nLabel distribution:')
print(data['label'].value_counts())

# Show data correlation
numeric_features = data.select_dtypes(include=['float64', 'int64'])
corr_matrix = numeric_features.corr()
feature_numbers = range(1, len(corr_matrix.columns) + 1)
corr_matrix.columns = feature_numbers
corr_matrix.index = feature_numbers
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title('Correlation Heatmap (All Features)')
plt.show()

# Normality and outlier check
print('\nNormality and Outlier Check:')

if 'label' in numeric_features.columns:
    numeric_features = numeric_features.drop('label', axis=1)

normal_count = 0
outlier_counts = {}

for col in numeric_features.columns:
    # Drop missing values for the statistical test
    col_data = numeric_features[col].dropna()
    
    # 1. Normality Test (Shapiro-Wilk)
    stat, p = stats.shapiro(col_data)
    if p > 0.05:
        normal_count += 1
        
    # 2. Outlier Detection (IQR Method)
    Q1 = col_data.quantile(0.25)
    Q3 = col_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
    outlier_counts[col] = len(outliers)

# Summarize Normality
total_features = len(numeric_features.columns)
print(f"Normally distributed features (Shapiro-Wilk p > 0.05): {normal_count} out of {total_features} ({round((normal_count/total_features)*100, 1)}%)")

# Summarize Outliers
total_outliers = sum(outlier_counts.values())
features_with_outliers = sum(1 for count in outlier_counts.values() if count > 0)
print(f"Total outliers detected across all features: {total_outliers}")
print(f"Features containing at least one outlier: {features_with_outliers} out of {total_features}")

# Plot a boxplot for a quick visual check of the first 10 features
plt.figure(figsize=(12, 6))
sns.boxplot(data=numeric_features.iloc[:, :10])
plt.title('Boxplot of the first 10 numeric features (Visualizing Outliers)')
plt.xticks(rotation=45)
plt.show()

# %% Part 2: Data Preprocessing`

# Part 2.1: Custom transformers
# Correlation filter for removing highly correlated features
class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.90):
        self.threshold = threshold
        self.to_drop = []

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        corr_matrix = X_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop = [column for column in upper.columns if any(upper[column] > self.threshold)]
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        return X_df.drop(columns=self.to_drop)

#Winsorizer for handling extreme outliers
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, limits=(0.01, 0.01)):
        self.limits = limits
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        lower_quantile = self.limits[0]
        upper_quantile = 1.0 - self.limits[1]
        self.lower_bounds_ = X_df.quantile(lower_quantile)
        self.upper_bounds_ = X_df.quantile(upper_quantile)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        X_winsorized = X_df.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)
        return X_winsorized

#Mann-Whitney U test for feature selection
class MannWhitneyFilter(BaseEstimator, TransformerMixin):
    def __init__(self, n_features_to_select=10):
        self.n_features_to_select = n_features_to_select
        self.selected_indices_ = []

    def fit(self, X, y):
        X_df = pd.DataFrame(X)
        y_arr = np.array(y)
        class_0 = X_df[y_arr == 0]
        class_1 = X_df[y_arr == 1]
        p_values = []

        for col in X_df.columns:
            stat, p_val = mannwhitneyu(class_0[col], class_1[col], alternative='two-sided')
            p_values.append((col, p_val))

        p_values.sort(key=lambda x: x[1])
        k = min(self.n_features_to_select, len(X_df.columns))
        self.selected_indices_ = [item[0] for item in p_values[:k]]
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        return X_df[self.selected_indices_]
    
# Part 2.2: Dictionaries & distributions
# Label encoding
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])
print(f'\nLabel encoding mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}')

X = data.drop(['label'], axis=1)
y = data['label']

# Define feature selectors
feature_selectors = {
    'Mann-Whitney U-Test': MannWhitneyFilter(),
    'LASSO': SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', random_state=42)),
    'RFE_LR': RFE(estimator=LogisticRegression(random_state=42), step=1),
    'None': 'passthrough'
}

# Define classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Define parameter distributions for RandomizedSearchCV
param_distributions = [
    # Logistic Regression
    {
        'feature_selection': [feature_selectors['LASSO']],
        'feature_selection__estimator__C': loguniform(0.01, 10),
        'feature_selection__max_features': [25],
        'classifier': [classifiers['Logistic Regression']],
        'classifier__C': loguniform(0.001, 100),
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear']
    },
    {
        'feature_selection': [feature_selectors['RFE_LR']],
        'feature_selection__n_features_to_select': randint(5, 25),
        'classifier': [classifiers['Logistic Regression']],
        'classifier__C': loguniform(0.001, 100),
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear']
    },
    {
        'feature_selection': [feature_selectors['Mann-Whitney U-Test']],
        'feature_selection__n_features_to_select': randint(5, 25),
        'classifier': [classifiers['Logistic Regression']],
        'classifier__C': loguniform(0.01, 10),
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear']
    },
    {
        'feature_selection': [feature_selectors['None']],
        'classifier': [classifiers['Logistic Regression']],
        'classifier__C': loguniform(0.01, 10),
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear']
    },

    # Random Forest
    {
        'feature_selection': [feature_selectors['LASSO']],
        'feature_selection__estimator__C': loguniform(0.01, 10),
        'feature_selection__max_features': [25],
        'classifier': [classifiers['Random Forest']],
        'classifier__n_estimators': randint(50, 200),
        'classifier__max_depth': [2, 5, 10],
        'classifier__max_features': ['sqrt', 'log2', 0.2, 0.4]
    },
    {
        'feature_selection': [feature_selectors['RFE_LR']],
        'feature_selection__n_features_to_select': randint(5, 25),
        'classifier': [classifiers['Random Forest']],
        'classifier__n_estimators': randint(50, 200),
        'classifier__max_depth': [2, 5, 10],
        'classifier__max_features': ['sqrt', 'log2', 0.2, 0.4]
    },
    {
        'feature_selection': [feature_selectors['Mann-Whitney U-Test']],
        'feature_selection__n_features_to_select': randint(5, 25),
        'classifier': [classifiers['Random Forest']],
        'classifier__n_estimators': randint(50, 200),
        'classifier__max_depth': [2, 5, 10],
        'classifier__max_features': ['sqrt', 'log2', 0.2, 0.4]
    },
    {
        'feature_selection': [feature_selectors['None']],
        'classifier': [classifiers['Random Forest']],
        'classifier__n_estimators': randint(50, 200),
        'classifier__max_depth': [2, 5, 10],
        'classifier__max_features': ['sqrt', 'log2', 0.2, 0.4]
    },

    # SVM
    {
        'feature_selection': [feature_selectors['LASSO']],
        'feature_selection__estimator__C': loguniform(0.01, 10),
        'feature_selection__max_features': [25],
        'classifier': [classifiers['SVM']],
        'classifier__C': loguniform(0.01, 100),
        'classifier__kernel': ['linear', 'rbf']
    },
    {
        'feature_selection': [feature_selectors['RFE_LR']],
        'feature_selection__n_features_to_select': randint(5, 25),
        'classifier': [classifiers['SVM']],
        'classifier__C': loguniform(0.01, 100),
        'classifier__kernel': ['linear', 'rbf']
    },
    {
        'feature_selection': [feature_selectors['Mann-Whitney U-Test']],
        'feature_selection__n_features_to_select': randint(5, 25),
        'classifier': [classifiers['SVM']],
        'classifier__C': loguniform(0.01, 100),
        'classifier__kernel': ['linear', 'rbf']
    },
    {
        'feature_selection': [feature_selectors['None']],
        'classifier': [classifiers['SVM']],
        'classifier__C': loguniform(0.01, 100),
        'classifier__kernel': ['linear', 'rbf']
    },

    # XGBoost
    {
        'feature_selection': [feature_selectors['LASSO']],
        'feature_selection__estimator__C': loguniform(0.01, 10),
        'feature_selection__max_features': [25],
        'classifier': [classifiers['XGBoost']],
        'classifier__n_estimators': randint(50, 200),
        'classifier__max_depth': [2, 3, 4],
        'classifier__learning_rate': loguniform(0.01, 0.2),
        'classifier__subsample': [0.6,0.8, 1.0]

        
    },
    {
        'feature_selection': [feature_selectors['RFE_LR']],
        'feature_selection__n_features_to_select': randint(5, 25),
        'classifier': [classifiers['XGBoost']],
        'classifier__n_estimators': randint(50, 200),
        'classifier__max_depth': [2, 3, 4],
        'classifier__learning_rate': loguniform(0.01, 0.2),
        'classifier__subsample': [0.6, 0.8, 1.0]
    },
    {
        'feature_selection': [feature_selectors['Mann-Whitney U-Test']],
        'feature_selection__n_features_to_select': randint(5, 25),
        'classifier': [classifiers['XGBoost']],
        'classifier__n_estimators': randint(50, 200),
        'classifier__max_depth': [2, 3, 4],
        'classifier__learning_rate': loguniform(0.01, 0.2),
        'classifier__subsample': [0.6,0.8, 1.0]
    },
    {
        'feature_selection': [feature_selectors['None']],
        'classifier': [classifiers['XGBoost']],
        'classifier__n_estimators': randint(50, 200),
        'classifier__max_depth': [2, 3, 4],
        'classifier__learning_rate': loguniform(0.01, 0.2),
        'classifier__subsample': [0.6,0.8, 1.0]
    }
]

# Part 2.4 Pipeline & search setup
# Pipeline setup
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median').set_output(transform="pandas")),       # Impute missing values with median
    ('variance_thresh', VarianceThreshold(threshold=0).set_output(transform="pandas")), # Remove features with zero variance
    ('scaler', RobustScaler().set_output(transform="pandas")),                          # Scale features with RobustScaler
    ('winsorizer', Winsorizer(limits=(0.01, 0.01))),                                    # Winsorize outliers       
    ('corr_filter', CorrelationFilter(threshold=0.90)),                                 # Remove highly correlated features
    ('feature_selection', 'passthrough'),                                               # Feature selection
    ('classifier', 'passthrough')                                                       # Classifier 
])

# Nested Cross-validation setup
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Number of random combinations to test
N_ITER = 15

# RandomizedSearchCV setup
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=N_ITER,
    cv=inner_cv,
    scoring=['accuracy', 'roc_auc'],
    refit='roc_auc',
    n_jobs=-1,
    verbose=1,
    random_state=42
)
# %% Part 3: Model training
#Nested Cross-validation with RandomizedSearchCV
print("\nStarting Nested Cross-validation with RandomizedSearchCV...")

# Store metrics for plotting and the final table
fold_results = []
tprs = []
aucs = []
all_y_test = []
all_y_pred = []
mean_fpr = np.linspace(0, 1, 100)

# Setup the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Execute the outer loop of nested cross-validation
for fold, (train_idx, test_idx) in enumerate(tqdm(outer_cv.split(X, y), total=outer_cv.get_n_splits(), desc="Outer CV Folds")):
    
    # Split the data for this specific fold
    X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
    y_train_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]
    
    # Run the inner RandomizedSearchCV to find the best model for this training fold
    random_search.fit(X_train_outer, y_train_outer)
    
    # Extract the best model
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    
    # Evaluate the best model on the hidden outer test fold
    y_pred = best_model.predict(X_test_outer)
    y_prob = best_model.predict_proba(X_test_outer)[:, 1]
    
    # Calculate fold metrics
    fold_acc = accuracy_score(y_test_outer, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test_outer, y_prob)
    fold_auc = auc(fpr, tpr)
    fold_f1 = f1_score(y_test_outer, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test_outer, y_pred).ravel()
    fold_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fold_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Calculate features remaining after VarianceThreshold and CorrelationFilter
    var_thresh_step = best_model.named_steps['variance_thresh']
    features_after_var = sum(var_thresh_step.get_support())
    corr_filter_step = best_model.named_steps['corr_filter']
    features_after_corr = features_after_var - len(corr_filter_step.to_drop)

    # Extract the final number of features used by the classifier
    fold_num_features = best_model.named_steps['classifier'].n_features_in_

    # Save true and predicted labels for the cumulative confusion matrix
    all_y_test.extend(y_test_outer)
    all_y_pred.extend(y_pred)
    
    # Save the ROC curve data
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(fold_auc)
    
    # Plot the ROC curve
    ax.plot(fpr, tpr, alpha=0.3, label=f'ROC fold {fold+1} (AUC = {fold_auc:.2f})')
    
    # Clean up names for the table
    classifier_name = str(best_model.named_steps['classifier'].__class__.__name__)
    feature_selector_name = str(best_model.named_steps['feature_selection'].__class__.__name__)
    if feature_selector_name == "str": 
        feature_selector_name = "None"
        
    # Store the results for the summary table
    fold_results.append({
        'Fold': fold + 1,
        'Classifier': classifier_name,
        'Feature Selector': feature_selector_name,
        'After VarThresh': features_after_var,
        'After CorrFilter': features_after_corr,
        'Final Features': fold_num_features,
        'Outer Accuracy': round(fold_acc, 4),
        'Outer AUC': round(fold_auc, 4),
        'Outer F1': round(fold_f1, 4),
        'Outer Sensitivity': round(fold_sensitivity, 4),
        'Outer Specificity': round(fold_specificity, 4),
        'Best Params (Raw)': best_params
    })

# Part 4: Final evaluation and plotting
# Plot the Mean ROC curve
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=0.8)

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance (AUC = 0.50)', alpha=0.8)

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Receiver Operating Characteristic (Nested CV)")
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc="lower right")
plt.show()

# Plot the cumulative confusion matrix
print("\n--- Cumulative Confusion Matrix (All Folds Combined) ---")
cm = confusion_matrix(all_y_test, all_y_pred)
fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues, ax=ax_cm, values_format='d')
plt.title("Confusion Matrix (Nested CV Combined)")
plt.show()

# Plot the summary table
print("\n--- Summary of Best Models per Outer Fold ---")
results_df = pd.DataFrame(fold_results)
print(results_df.drop(columns=['Best Params (Raw)']))

# Show the winning hyperparameters for each fold
print("\n--- Winning Hyperparameters per Fold ---")
for index, row in results_df.iterrows():
    print(f"\nFold {row['Fold']} ({row['Final Features']} features): {row['Classifier']} + {row['Feature Selector']}")
    for param_name, param_value in row['Best Params (Raw)'].items():
        if hasattr(param_value, '__class__') and not isinstance(param_value, (int, float, str, bool, type(None))):
            print(f"   - {param_name}: {param_value.__class__.__name__}")
        elif isinstance(param_value, float):
            print(f"   - {param_name}: {param_value:.4f}")
        else:
            print(f"   - {param_name}: {param_value}")

#Show the unbiased overall metrics
print("\n--- Unbiased Overall Performance ---")
print(f"Mean Features after VarThresh: {results_df['After VarThresh'].mean():.1f} (+/- {results_df['After VarThresh'].std():.1f})")
print(f"Mean Features after CorrFilter: {results_df['After CorrFilter'].mean():.1f} (+/- {results_df['After CorrFilter'].std():.1f})")
print(f"Mean Final Features Used:      {results_df['Final Features'].mean():.1f} (+/- {results_df['Final Features'].std():.1f})")
print(f"Mean Nested Accuracy:          {results_df['Outer Accuracy'].mean():.4f} (+/- {results_df['Outer Accuracy'].std():.4f})")
print(f"Mean Nested AUC:               {results_df['Outer AUC'].mean():.4f} (+/- {results_df['Outer AUC'].std():.4f})")
print(f"Mean Nested F1-Score:          {results_df['Outer F1'].mean():.4f} (+/- {results_df['Outer F1'].std():.4f})")
print(f"Mean Nested Sensitivity:       {results_df['Outer Sensitivity'].mean():.4f} (+/- {results_df['Outer Sensitivity'].std():.4f})")
print(f"Mean Nested Specificity:       {results_df['Outer Specificity'].mean():.4f} (+/- {results_df['Outer Specificity'].std():.4f})")


#%% Part 5: Train the final model on the entire dataset
# Final search on the entire dataset
final_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=N_ITER,
    cv=inner_cv,
    scoring=['accuracy', 'roc_auc'],
    refit='roc_auc',
    n_jobs=-1,
    random_state=42
)

final_search.fit(X, y)

# Final model
final_model = final_search.best_estimator_
print(f"\nFinal chosen classifier: {final_model.named_steps['classifier'].__class__.__name__}")
print(f"Final chosen feature selector: {final_model.named_steps['feature_selection'].__class__.__name__}")
print(f"\n--- Final Model Hyperparameters ---")
final_classifier_name = str(final_model.named_steps['classifier'].__class__.__name__)
final_selector_name = str(final_model.named_steps['feature_selection'].__class__.__name__)
if final_selector_name == "str": 
    final_selector_name = "None"
print(f"Model: {final_classifier_name} + {final_selector_name}")
for param_name, param_value in final_search.best_params_.items():
    if hasattr(param_value, '__class__') and not isinstance(param_value, (int, float, str, bool, type(None))):
        print(f"   - {param_name}: {param_value.__class__.__name__}")
    elif isinstance(param_value, float):
        print(f"   - {param_name}: {param_value:.4f}")
    else:
        print(f"   - {param_name}: {param_value}")

# %%

