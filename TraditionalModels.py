import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE

# Load dataset with optimized memory and dtype handling
data = pd.read_csv("creditcard.csv", low_memory=False)
data = data.apply(lambda col: pd.to_numeric(col, errors='coerce') if col.dtype == 'object' else col)
data = data.dropna()

# Separate features and target variable
X = data.drop("Class", axis=1)
y = data["Class"]

# Ensure no NaNs in target variable
X = X[~y.isna()]
y = y.dropna()

# Train-test split with stratification
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_index, test_index = next(sss.split(X, y))
X_train, X_test = X.iloc[train_index], X.iloc[test_index]
y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Apply SMOTE for class balancing
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Optional: Dimensionality Reduction with PCA
pca = PCA(n_components=0.95, random_state=42)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Define parameter grids for RandomizedSearchCV for each model
rf_params = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20, 30]}
logreg_params = {'C': np.logspace(-3, 3, 10), 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}
svm_params = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
knn_params = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}

# Function to perform RandomizedSearchCV and output results
def model_training(X_train, y_train, X_test, y_test, model, params, model_name):
    random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=5, scoring='f1', cv=3, n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    # Predict and evaluate
    y_pred = best_model.predict(X_test)
    print(f"{model_name} Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"{model_name} Classification Report:\n", classification_report(y_test, y_pred))

# Train and evaluate each model
print("Training Random Forest...")
model_training(X_train, y_train, X_test, y_test, RandomForestClassifier(), rf_params, "Random Forest")

print("Training Logistic Regression...")
model_training(X_train, y_train, X_test, y_test, LogisticRegression(), logreg_params, "Logistic Regression")

print("Training Support Vector Machine...")
model_training(X_train, y_train, X_test, y_test, SVC(), svm_params, "Support Vector Machine")

print("Training K-Nearest Neighbors...")
model_training(X_train, y_train, X_test, y_test, KNeighborsClassifier(), knn_params, "K-Nearest Neighbors")
