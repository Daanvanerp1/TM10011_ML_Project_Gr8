import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, RocCurveDisplay
from TestjesDaan.feature_selection import X_train_mrmr, X_test_mrmr, y_train, y_test

# --- STAP 1: Modellen Initialiseren ---
# We gebruiken 'class_weight="balanced"' omdat jullie mogelijk ongelijke 
# aantallen GIST en NON-GIST patiënten hebben. Dit voorkomt dat het model 
# een voorkeur krijgt voor de grootste groep.

lr_model = LogisticRegression(class_weight='balanced', random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
# Tip: max_depth=5 bij de Random Forest voorkomt overfitting op jullie kleine dataset!

# --- STAP 2: Modellen Trainen (Fit) ---
print("Modellen aan het trainen op de mRMR features...")
lr_model.fit(X_train_mrmr, y_train)
rf_model.fit(X_train_mrmr, y_train)

# --- STAP 3: Voorspellingen doen op de TESTSET ---
# Voor de ROC-AUC hebben we niet de harde labels (0 of 1) nodig, 
# maar de onzekerheid/kans (probabilities tussen 0.0 en 1.0).
lr_probs = lr_model.predict_proba(X_test_mrmr)[:, 1]
rf_probs = rf_model.predict_proba(X_test_mrmr)[:, 1]

# Voor het classification report hebben we wél de harde labels nodig
lr_preds = lr_model.predict(X_test_mrmr)
rf_preds = rf_model.predict(X_test_mrmr)

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
RocCurveDisplay.from_estimator(lr_model, X_test_mrmr, y_test, name="Logistic Regression", ax=ax)
RocCurveDisplay.from_estimator(rf_model, X_test_mrmr, y_test, name="Random Forest", ax=ax)

# Maak de grafiek mooi voor jullie uiteindelijke verslag/presentatie
ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Gokken (AUC = 0.50)')
plt.title("ROC Curves: GIST vs NON-GIST Voorspelling", fontsize=14)
plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()