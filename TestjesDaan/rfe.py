import numpy as np
import pandas as pd
from TestjesDaan.feature_selection import X_test_mrmr
from worcgist.load_data import load_data
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, RocCurveDisplay


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

# --- STAP 1: Bepaal het gewenste aantal features ---
AANTAL_FEATURES = 10 
print(f"Starten van RFE om terug te gaan van {X_train_final.shape[1]} naar {AANTAL_FEATURES} features...")

# --- STAP 2: Kies het 'Jury' model ---
# We gebruiken een Random Forest. (max_depth=5 voorkomt dat de jury overfit tijdens het kiezen)
# Alternatief: je kunt hier ook LogisticRegression(...) gebruiken als jury.
jury_model = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)

# --- STAP 3: Configureer en Train RFE ---
# step=1 betekent dat hij bij elke ronde streng 1 feature weggooit en opnieuw traint.
# (Dit is heel precies, maar duurt iets langer. Als het te traag is, zet step op 2 of 5).
rfe_selector = RFE(estimator=jury_model, n_features_to_select=AANTAL_FEATURES, step=1)

# Fit de RFE op de schone data
rfe_selector.fit(X_train_final, y_train)

# --- STAP 4: Haal de geselecteerde features op ---
# rfe_selector.support_ geeft een lijst met True/False voor de overlevende features
rfe_mask = rfe_selector.support_
geselecteerde_features_rfe = X_train_final.columns[rfe_mask]

# Als je ook de 'ranking' wilt zien van de weggegooide features:
# rfe_selector.ranking_ (1 = geselecteerd, 2 = als laatste afgevallen, etc.)

# --- STAP 5: Filter de datasets ---
X_train_rfe = X_train_final[geselecteerde_features_rfe]
X_test_rfe = X_test_final[geselecteerde_features_rfe]

# --- Samenvatting printen ---
print("\n" + "="*40)
print("🏆 RFE RESULTATEN")
print("="*40)
print("Dit is jullie RFE radiomics signatuur:")
for i, feature in enumerate(geselecteerde_features_rfe):
    print(f"{i+1}. {feature}")



# --- STAP 1: Modellen Initialiseren ---
# We gebruiken 'class_weight="balanced"' omdat jullie mogelijk ongelijke 
# aantallen GIST en NON-GIST patiënten hebben. Dit voorkomt dat het model 
# een voorkeur krijgt voor de grootste groep.

lr_model = LogisticRegression(class_weight='balanced', random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
# Tip: max_depth=5 bij de Random Forest voorkomt overfitting op jullie kleine dataset!

# --- STAP 2: Modellen Trainen (Fit) ---
print("Modellen aan het trainen op de mRMR features...")
lr_model.fit(X_train_rfe, y_train)
rf_model.fit(X_train_rfe, y_train)

# --- STAP 3: Voorspellingen doen op de TESTSET ---
# Voor de ROC-AUC hebben we niet de harde labels (0 of 1) nodig, 
# maar de onzekerheid/kans (probabilities tussen 0.0 en 1.0).
lr_probs = lr_model.predict_proba(X_test_rfe)[:, 1]
rf_probs = rf_model.predict_proba(X_test_rfe)[:, 1]

# Voor het classification report hebben we wél de harde labels nodig
lr_preds = lr_model.predict(X_test_rfe)
rf_preds = rf_model.predict(X_test_rfe)

# --- STAP 4: Evaluatie & Metrics ---
print("\n" + "="*40)
print("🏆 LOGISTIC REGRESSION RESULTATEN")
print("="*40)
print(f"ROC-AUC Score: {roc_auc_score(y_test, lr_probs):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, lr_preds, target_names=['NON-GIST (0)', 'GIST (1)']))

print("\n" + "="*40)
print("🌲 RANDOM FOREST RESULTATEN")
print("="*40)
print(f"ROC-AUC Score: {roc_auc_score(y_test, rf_probs):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_preds, target_names=['NON-GIST (0)', 'GIST (1)']))

# --- STAP 5: ROC Curve Visualisatie ---
fig, ax = plt.subplots(figsize=(8, 6))

# Plot de curves voor beide modellen in één grafiek
RocCurveDisplay.from_estimator(lr_model, X_test_rfe, y_test, name="Logistic Regression", ax=ax)
RocCurveDisplay.from_estimator(rf_model, X_test_rfe, y_test, name="Random Forest", ax=ax)

# Maak de grafiek mooi voor jullie uiteindelijke verslag/presentatie
ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Gokken (AUC = 0.50)')
plt.title("ROC Curves: GIST vs NON-GIST Voorspelling", fontsize=14)
plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()