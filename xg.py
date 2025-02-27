import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, accuracy_score, f1_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the dataset
df = pd.read_csv('creditcard.csv')

# Preprocessing
# Check for null values (no action here since the dataset is clean)
print(f"Null values:\n{df.isnull().sum()}")

# Standardize the 'Amount' column
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# Split data into features and target
X = df.drop(['Class'], axis=1)
y = df['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model and Hyperparameter Tuning
xgb = XGBClassifier(max_depth=2, n_estimators=200)
param_grid = {
    'learning_rate': [0.2, 0.6],
    'subsample': [0.3, 0.6, 0.9]
}
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='roc_auc', cv=3, verbose=1)
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)

# Train the model with the best parameters
best_xgb = grid_search.best_estimator_
best_xgb.fit(X_train, y_train)

# Save the trained model
with open('xgb_model.pkl', 'wb') as file:
    pickle.dump(best_xgb, file)

print("Model saved successfully as 'xgb_model.pkl'")

# Evaluate the model on the test set
y_test_pred = best_xgb.predict(X_test)
y_test_pred_proba = best_xgb.predict_proba(X_test)[:, 1]

# Metrics
print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_test_pred)
print(conf_matrix)

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

auc = roc_auc_score(y_test, y_test_pred_proba)
print(f"ROC AUC: {auc:.2f}")

# ROC Curve
def draw_roc(actual, probs):
    fpr, tpr, thresholds = roc_curve(actual, probs)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

draw_roc(y_test, y_test_pred_proba)
