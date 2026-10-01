# Necessary libraries install/import karein
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 1: Dataset ko load aur understand karein
iris = load_iris()
X = iris.data  # 4 Features: Sepal Length, Sepal Width, Petal Length, Petal Width
y = iris.target  # Target Labels: Setosa (0), Versicolor (1), Virginica (2)

print("--- Data Overview ---")
print("Feature Names:", iris.feature_names)
print("Target Classes:", iris.target_names)
print("Total Samples:", X.shape[0])

# Step 2: Data ko Shuffle aur Split karein (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Step 3: Feature Scaling (StandardScaler - Mean=0, Variance=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: K-Nearest Neighbors (KNN) Classifier Train karein (K=5)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

# Step 5: Test set par Prediction karein
y_pred = knn_model.predict(X_test_scaled)

# Step 6: Output Evaluation (Confusion Matrix, Precision, Recall, F1-Score)
print("\n--- Model Evaluation ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report (Precision, Recall, F1-Score):")
print(classification_report(y_test, y_pred, target_names=iris.target_names))