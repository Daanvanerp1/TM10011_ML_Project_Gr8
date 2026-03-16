
#%% 
import numpy as np
import pandas as pd
from worcgist.load_data import load_data
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold



data = load_data()
# Label encoding (Switch Gist and non-Gist to 1 and 0 respectively)
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# Split data into training and testing sets, stratify and use 20% of the data for testing
X = data.drop('label', axis=1)
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- STAP 1: Schalen (RobustScaler) ---
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- STAP 2: Variantie Filter ---
# Tip: Speel met deze drempel. Bij RobustScaler is de variantie vaak klein.
selector = VarianceThreshold(threshold=0.01)
X_train_selected = selector.fit_transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)

# --- STAP 3: Winsorization (Veilig tegen Data Leakage!) ---
# Bereken grenzen ALLEEN op trainingsset
lower_limits = np.percentile(X_train_selected, 5, axis=0)
upper_limits = np.percentile(X_train_selected, 95, axis=0)

# Knip de extremen af voor train en test
X_train_winsorized = np.clip(X_train_selected, lower_limits, upper_limits)
X_test_winsorized = np.clip(X_test_selected, lower_limits, upper_limits)

# --- STAP 4: Correlatie Filter (Spearman > 0.90) ---
# Zet de numpy arrays even om naar Pandas DataFrames voor de berekening
df_train = pd.DataFrame(X_train_winsorized)
df_test = pd.DataFrame(X_test_winsorized)

# Bereken Spearman correlatiematrix ALLEEN op de trainingsset
corr_matrix = df_train.corr(method='spearman').abs()

# Pak de bovenste driehoek om dubbele verwijderingen te voorkomen
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Zoek de indexen van de kolommen die te sterk correleren
correlatie_drempel = 0.90
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > correlatie_drempel)]

# Gooi deze kolommen weg uit ZOWEL train als test
X_train_final = df_train.drop(columns=to_drop)
X_test_final = df_test.drop(columns=to_drop)

# --- Samenvatting printen ---
print(f"1. Origineel aantal features: {X_train.shape[1]}")
print(f"2. Features over na Variantie filter: {X_train_selected.shape[1]}")
print(f"3. Aantal features verwijderd door hoge correlatie (> {correlatie_drempel}): {len(to_drop)}")
print(f"4. DEFINITIEF aantal features voor modelbouw: {X_train_final.shape[1]}")
# %%
from sklearn.linear_model import LogisticRegressionCV

# We gaan ervan uit dat jullie y_train en y_test al hebben 
# (bijv. 0 = NON-GIST, 1 = GIST)

print(f"Starten van LASSO met {X_train_final.shape[1]} features...")

# --- STAP 1: Configureer en Train het LASSO model ---
# penalty='l1' zorgt voor de LASSO selectie (krimpt coëfficiënten naar nul)
# solver='liblinear' of 'saga' is vereist voor L1 penalties
# cv=5 betekent 5-voudige kruisvalidatie om de perfecte straf (C) te vinden
# max_iter=10000 voorkomt foutmeldingen als het model moeite heeft met convergeren
lasso_cv = LogisticRegressionCV(
    Cs=10,                  # Test 10 verschillende straf-waardes
    cv=5,                   # 5-fold cross-validation
    penalty='l1',           # Dit is de magie van LASSO
    solver='liblinear',     # Standaard solver voor kleine datasets met l1
    random_state=42,        # Voor reproduceerbaarheid (zodat jullie steeds dezelfde output krijgen)
    max_iter=10000,
    class_weight='balanced' # Heel belangrijk als er ongelijke aantallen GIST/NON-GIST zijn!
)

# Fit het model uitsluitend op de trainingsset
lasso_cv.fit(X_train_final, y_train)

# --- STAP 2: Haal de overlevende features eruit ---
# Kijk welke features een coëfficiënt hebben die NIET nul is
lasso_coefs = lasso_cv.coef_[0]
selected_features_mask = lasso_coefs != 0

# Haal de namen van de geselecteerde kolommen op
selected_columns = X_train_final.columns[selected_features_mask]

# --- STAP 3: Filter de datasets ---
X_train_lasso = X_train_final[selected_columns]
X_test_lasso = X_test_final[selected_columns]

# --- Samenvatting printen ---
print("\n--- LASSO Resultaten ---")
print(f"De optimale hyperparameter (C) was: {lasso_cv.C_[0]:.4f}")
print(f"Aantal geselecteerde features: {len(selected_columns)}")
print("\nDit is jullie gouden radiomics signatuur:")
for i, col in enumerate(selected_columns):
    # Print de naam van de feature en het bijbehorende 'gewicht' in het model
    print(f"{i+1}. Feature {col} (Coëfficiënt: {lasso_coefs[selected_features_mask][i]:.4f})")
# %%

# Reset de index van de labels zodat ze weer matchen met de nieuwe X_train_final
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
# Importeer de classificatie-versie van mRMR (omdat GIST vs NON-GIST binair is)
from mrmr import mrmr_classif

# --- STAP 1: Bepaal het gewenste aantal features ---
# Gebaseerd op jullie 246 patiënten, is ergens tussen de 5 en 15 features optimaal.
AANTAL_FEATURES = 15 

print(f"Starten van mRMR selectie om de top {AANTAL_FEATURES} features te vinden uit de {X_train_final.shape[1]} overgebleven features...")

# --- STAP 2: Train het mRMR algoritme ---
# Let op: mRMR verwacht dat X_train_final een Pandas DataFrame is 
# en y_train een Pandas Series (of een 1D numpy array).
# Het algoritme berekent automatisch de F-statistiek (relevantie) en correlatie (redundantie).
geselecteerde_features = mrmr_classif(X=X_train_final, y=y_train, K=AANTAL_FEATURES)

# --- STAP 3: Filter de datasets ---
X_train_mrmr = X_train_final[geselecteerde_features]
X_test_mrmr = X_test_final[geselecteerde_features]

# --- Samenvatting printen ---
print("\n" + "="*40)
print("🏆 mRMR RESULTATEN")
print("="*40)
print("Dit is jullie alternatieve radiomics signatuur (in volgorde van belangrijkheid):")
for i, feature in enumerate(geselecteerde_features):
    print(f"{i+1}. {feature}")
# %%
