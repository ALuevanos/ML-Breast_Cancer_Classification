# -*- coding: utf-8 -*-
"""ML Breast Cancer Stage Prediction"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split, cross_val_score

# Load and preprocess the dataset
file_path = "Breast_Cancer (1).csv"  # Replace with your dataset path
data = pd.read_csv(file_path)

# Standardize column names and clean missing values
data.columns = data.columns.str.strip().str.lower()
data = data.dropna()

# Define target and feature variables
target_column = "6th stage"  # The cancer stage is the target
X = data.drop(columns=[target_column])
y = data[target_column]

# categorical variables
label_encoders = {}
for column in X.select_dtypes(include="object").columns:
    label_encoders[column] = LabelEncoder()
    X[column] = label_encoders[column].fit_transform(X[column])

# the target variable
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Split dataset into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Random Forest Classifier
rf_model = RandomForestClassifier(class_weight="balanced", random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)

# Evaluate Random Forest
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_classification_rep = classification_report(
    y_test, y_pred_rf, target_names=target_encoder.classes_
)
rf_conf_matrix = confusion_matrix(y_test, y_pred_rf)
rf_cross_val = cross_val_score(rf_model, X, y, cv=5, scoring="accuracy")

print("\nRandom Forest Evaluation:")
print("Accuracy:", rf_accuracy)
print("Classification Report:\n", rf_classification_rep)
print("Confusion Matrix:\n", rf_conf_matrix)
print("Cross-Validation Scores:", rf_cross_val)
print("Mean Accuracy:", rf_cross_val.mean())
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba_rf, multi_class="ovr"))

# Feature Importances
feature_importances = rf_model.feature_importances_
sorted_features = sorted(
    zip(X.columns, feature_importances), key=lambda x: x[1], reverse=True
)
print("\nFeature Importances:")
for feature, importance in sorted_features:
    print(f"{feature}: {importance:.4f}")

# SVM Classifier
svm_model = SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced")
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
y_proba_svm = svm_model.predict_proba(X_test)
svm_cross_val = cross_val_score(svm_model, X, y, cv=5, scoring="accuracy")

# Evaluate SVM
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_classification_rep = classification_report(
    y_test, y_pred_svm, target_names=target_encoder.classes_
)
svm_conf_matrix = confusion_matrix(y_test, y_pred_svm)

print("\nSVM Evaluation:")
print("Accuracy:", svm_accuracy)
print("Classification Report:\n", svm_classification_rep)
print("Confusion Matrix:\n", svm_conf_matrix)
print("Cross-Validation Scores:", svm_cross_val)
print("Mean Accuracy:", svm_cross_val.mean())
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba_svm, multi_class="ovr"))

# Decision Tree Visualization
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(X, y)
plt.figure(figsize=(15, 10))
plot_tree(
    tree_model,
    feature_names=X.columns,
    class_names=[f"Stage {i}" for i in sorted(set(y))],
    filled=True,
    impurity=True,
    fontsize=12,
)
plt.title("Simplified Decision Tree (Breast Cancer Dataset)", fontsize=16)
plt.show()

# Feature Importance Plot
def plot_feature_importances(features, importances):
    sorted_idx = np.argsort(importances)[-10:]
    plt.barh([features[i] for i in sorted_idx], importances[sorted_idx], color="skyblue")
    plt.xlabel("Importance")
    plt.title("Feature Importances")
    plt.show()

plot_feature_importances(X.columns, rf_model.feature_importances_)

# Confusion Matrix Plot
def plot_confusion_matrix(y_true, y_pred, labels):
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, cmap=plt.cm.Blues
    )
    plt.title("Confusion Matrix")
    plt.show()

plot_confusion_matrix(y_test, y_pred_rf, target_encoder.classes_)

# Class Distribution
class_distribution = pd.Series(y).value_counts()
print("\nClass Distribution:\n", class_distribution)
sns.countplot(x=y, palette="viridis")
plt.title("Class Distribution of Breast Cancer Stages")
plt.xlabel("Cancer Stage (Encoded)")
plt.ylabel("Number of Instances")
plt.show()

# ROC Curve for SVM
classes = np.unique(y_test)
y_test_binarized = label_binarize(y_test, classes=classes)

plt.figure(figsize=(10, 6))
for i, class_label in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_proba_svm[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {class_label} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Random Guessing")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("SVM ROC Curve (Multiclass)")
plt.legend(loc="lower right")
plt.show()
