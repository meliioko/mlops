import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score


data = pd.read_csv('tumors.csv')

# Définir les caractéristiques (X) et la cible (y)
X = data[['size', 'p53_concentration']]
y = data['is_cancerous']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Min Max Scaler
scaler = MinMaxScaler()
X_train[['size', 'p53_concentration']] = scaler.fit_transform(X_train[['size', 'p53_concentration']])
X_test[['size', 'p53_concentration']] = scaler.fit_transform(X_test[['size', 'p53_concentration']])



# Logistique regression
model = LogisticRegression()
model.fit(X_train, y_train)


# Prédire les classes et les probabilités pour les ensembles d'entraînement et de test
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

y_train_proba = model.predict_proba(X_train)[:, 1]
y_test_proba = model.predict_proba(X_test)[:, 1]

# Calculer les métriques pour l'ensemble d'entraînement
print("Training Metrics:")
print("Accuracy:", accuracy_score(y_train, y_train_pred))
print("Recall:", recall_score(y_train, y_train_pred))
print("Precision:", precision_score(y_train, y_train_pred))
print("F1 Score:", f1_score(y_train, y_train_pred))
print("AUC-ROC:", roc_auc_score(y_train, y_train_proba))

# Calculer les métriques pour l'ensemble de test
print("\nTest Metrics:")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Recall:", recall_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred))
print("F1 Score:", f1_score(y_test, y_test_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_test_proba))

# Sauvegarder le modèle
joblib.dump(model, 'tumor_model.joblib')
joblib.dump(scaler, 'scaler.joblib')