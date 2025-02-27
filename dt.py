from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle

# Load your dataset
data = pd.read_csv('creditcard.csv')  # Replace with the correct path to your dataset

# Check column names to verify the target column name
print(data.columns)

# Assuming the target column is 'Class' (replace if it's different)
X = data.drop('Class', axis=1)  # Features: drop the 'Class' column
y = data['Class']  # Target: the 'Class' column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Apply scaling (if required)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train_scaled, y_train)

# Save the trained model
with open('decision_tree_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")
