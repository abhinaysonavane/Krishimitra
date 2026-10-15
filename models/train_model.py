import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns

# 📁 Create folder if not exists
os.makedirs("saved", exist_ok=True)

# Load dataset
data = pd.read_csv('../realistic_crop_irrigation_dataset.csv')
data.columns = data.columns.str.strip()

# Features & Labels
X = data[['Temperature', 'Humidity', 'SoilMoisture']]
y_crop = data['Crop']
y_irrigation = data['Irrigation_Needed']

# Encode labels
crop_encoder = LabelEncoder()
y_crop = crop_encoder.fit_transform(y_crop)

irrigation_encoder = LabelEncoder()
y_irrigation = irrigation_encoder.fit_transform(y_irrigation)

# Train-test split
X_train, X_test, y_crop_train, y_crop_test = train_test_split(
    X, y_crop, test_size=0.2, random_state=42
)

_, _, y_irrigation_train, y_irrigation_test = train_test_split(
    X, y_irrigation, test_size=0.2, random_state=42
)

# ===========================
# 🌾 CROP MODELS
# ===========================

rf_crop = RandomForestClassifier()
xgb_crop = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    eval_metric='mlogloss'
)

rf_crop.fit(X_train, y_crop_train)
xgb_crop.fit(X_train, y_crop_train)

# Predictions
rf_pred_crop = rf_crop.predict(X_test)
xgb_pred_crop = xgb_crop.predict(X_test)

# Accuracy
rf_acc_crop = accuracy_score(y_crop_test, rf_pred_crop)
xgb_acc_crop = accuracy_score(y_crop_test, xgb_pred_crop)

print("\n🌾 Crop Model Evaluation")
print("RF Accuracy:", rf_acc_crop)
print("XGB Accuracy:", xgb_acc_crop)

# Reports
print("\nRF Report:\n", classification_report(y_crop_test, rf_pred_crop))
print("\nXGB Report:\n", classification_report(y_crop_test, xgb_pred_crop))

# Confusion Matrix (XGB)
cm_crop = confusion_matrix(y_crop_test, xgb_pred_crop)
plt.figure()
sns.heatmap(cm_crop, annot=True, fmt='d')
plt.title("Crop Confusion Matrix (XGB)")
plt.savefig("saved/crop_confusion_matrix.png")
plt.close()

# Best model
best_crop = xgb_crop if xgb_acc_crop > rf_acc_crop else rf_crop

# ===========================
# 💧 IRRIGATION MODELS
# ===========================

rf_irrigation = RandomForestClassifier()
xgb_irrigation = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    eval_metric='logloss'
)

rf_irrigation.fit(X_train, y_irrigation_train)
xgb_irrigation.fit(X_train, y_irrigation_train)

# Predictions
rf_pred_irrig = rf_irrigation.predict(X_test)
xgb_pred_irrig = xgb_irrigation.predict(X_test)

# Accuracy
rf_acc_irrig = accuracy_score(y_irrigation_test, rf_pred_irrig)
xgb_acc_irrig = accuracy_score(y_irrigation_test, xgb_pred_irrig)

print("\n💧 Irrigation Model Evaluation")
print("RF Accuracy:", rf_acc_irrig)
print("XGB Accuracy:", xgb_acc_irrig)

# Reports
print("\nRF Report:\n", classification_report(y_irrigation_test, rf_pred_irrig))
print("\nXGB Report:\n", classification_report(y_irrigation_test, xgb_pred_irrig))

# Confusion Matrix (RF)
cm_irrig = confusion_matrix(y_irrigation_test, rf_pred_irrig)
plt.figure()
sns.heatmap(cm_irrig, annot=True, fmt='d')
plt.title("Irrigation Confusion Matrix (RF)")
plt.savefig("saved/irrigation_confusion_matrix.png")
plt.close()

# Best model
best_irrigation = xgb_irrigation if xgb_acc_irrig > rf_acc_irrig else rf_irrigation

# ===========================
# 📊 COMPARISON GRAPH
# ===========================

models = ['RF Crop', 'XGB Crop', 'RF Irrigation', 'XGB Irrigation']
accuracy = [rf_acc_crop, xgb_acc_crop, rf_acc_irrig, xgb_acc_irrig]

plt.figure()
plt.bar(models, accuracy)
plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.savefig("saved/model_comparison.png")
plt.close()

# ===========================
# 💾 SAVE MODELS
# ===========================

joblib.dump(best_crop, 'saved/crop_model.pkl')
joblib.dump(best_irrigation, 'saved/irrigation_model.pkl')
joblib.dump(crop_encoder, 'saved/crop_encoder.pkl')
joblib.dump(irrigation_encoder, 'saved/irrigation_encoder.pkl')

print("\n✅ Models trained and saved successfully!")