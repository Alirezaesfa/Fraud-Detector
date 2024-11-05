import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import numpy as np

# Load the dataset
data = pd.read_csv("creditcard.csv")

# Define features and target variable
X = data.drop('Class', axis=1)  # All columns except the target
y = data['Class']  # Target variable

# Fill NaNs with 0
data = data.fillna(0)

# Ensure no NaNs in the target variable
X = X[~y.isna()]
y = y.dropna()

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balance the dataset
X_majority = X_scaled[y == 0]
y_majority = y[y == 0]

X_minority = X_scaled[y == 1]
y_minority = y[y == 1]

# Upsample minority class
X_minority_upsampled, y_minority_upsampled = resample(X_minority, y_minority, 
replace=True,     # sample with replacement
n_samples=len(y_majority),    # to match majority class
random_state=42) # reproducible results

# Combine majority and upsampled minority
X_balanced = np.vstack((X_majority, X_minority_upsampled))
y_balanced = np.hstack((y_majority, y_minority_upsampled))

# Split the balanced data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Hyperparameter tuning for Random Forest
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
rf_grid = GridSearchCV(RandomForestClassifier(), rf_param_grid, cv=5)
rf_grid.fit(X_train, y_train)
print("Best parameters for Random Forest:", rf_grid.best_params_)

# Train and evaluate Random Forest
rf_model = rf_grid.best_estimator_
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# Hyperparameter tuning for Logistic Regression
lr_param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'max_iter': [100, 200, 300]
}
lr_grid = GridSearchCV(LogisticRegression(solver='lbfgs'), lr_param_grid, cv=5)
lr_grid.fit(X_train, y_train)
print("Best parameters for Logistic Regression:", lr_grid.best_params_)

# Train and evaluate Logistic Regression
lr_model = lr_grid.best_estimator_
y_pred_lr = lr_model.predict(X_test)
print("Logistic Regression Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

# Train and evaluate Support Vector Machine
svm_model = SVC()
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
print("Support Vector Machine Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))

# Train and evaluate K-Nearest Neighbors
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
print("K-Nearest Neighbors Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
