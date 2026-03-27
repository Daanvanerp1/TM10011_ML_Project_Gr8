# ## Data loading and cleaning
# Below are functions to load the dataset of your choice. After that, it is all up to you to create and evaluate a classification method. Beware, there may be missing values in these datasets. Good luck!
#%%
import matplotlib.pyplot as plt
import seaborn as sns
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






# %%
