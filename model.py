import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.utils import resample

# Load the dataset
data = pd.read_csv("creditcard.csv")

# Define features and target variable
X = data.drop('Class', axis=1)
y = data['Class']

# Fill NaN values with 0
data = data.fillna(0)

# Ensure no NaNs in the target variable
X = X[~y.isna()]
y = y.dropna()

# Handle class imbalance
X_majority = X[y == 0]
y_majority = y[y == 0]
X_minority = X[y == 1]
y_minority = y[y == 1]

# Upsample minority class
X_minority_upsampled, y_minority_upsampled = resample(X_minority, y_minority, 
                                                      replace=True,     
                                                      n_samples=len(y_majority),    
                                                      random_state=42) 

X = pd.concat([X_majority, X_minority_upsampled])
y = pd.concat([y_majority, y_minority_upsampled])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define models
rf_model = RandomForestClassifier(random_state=42)
logreg_model = LogisticRegression(max_iter=1000, random_state=42)
svm_model = SVC(random_state=42)
knn_model = KNeighborsClassifier()

# Define parameter grids for each model
rf_params = {'n_estimators': [50, 100], 'max_depth': [None, 10]}
logreg_params = {'C': np.logspace(-3, 3, 5), 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}
svm_params = {'C': [0.1, 1], 'gamma': ['scale']}
knn_params = {'n_neighbors': [3, 5], 'weights': ['uniform']}

# Perform RandomizedSearchCV for hyperparameter tuning
rf_random_search = RandomizedSearchCV(rf_model, rf_params, n_iter=5, cv=2, random_state=42, n_jobs=-1)
logreg_random_search = RandomizedSearchCV(logreg_model, logreg_params, n_iter=5, cv=2, random_state=42, n_jobs=-1)
svm_random_search = RandomizedSearchCV(svm_model, svm_params, n_iter=5, cv=2, random_state=42, n_jobs=-1)
knn_random_search = RandomizedSearchCV(knn_model, knn_params, n_iter=5, cv=2, random_state=42, n_jobs=-1)

# Train models
rf_random_search.fit(X_train, y_train)
logreg_random_search.fit(X_train, y_train)
svm_random_search.fit(X_train, y_train)
knn_random_search.fit(X_train, y_train)

# Make predictions
rf_pred = rf_random_search.predict(X_test)
logreg_pred = logreg_random_search.predict(X_test)
svm_pred = svm_random_search.predict(X_test)
knn_pred = knn_random_search.predict(X_test)

# Generate classification reports
rf_report = classification_report(y_test, rf_pred, output_dict=True)
logreg_report = classification_report(y_test, logreg_pred, output_dict=True)
svm_report = classification_report(y_test, svm_pred, output_dict=True)
knn_report = classification_report(y_test, knn_pred, output_dict=True)

# Convert reports to DataFrames for saving
rf_report_df = pd.DataFrame(rf_report).transpose()
logreg_report_df = pd.DataFrame(logreg_report).transpose()
svm_report_df = pd.DataFrame(svm_report).transpose()
knn_report_df = pd.DataFrame(knn_report).transpose()

# Save reports to CSV files
rf_report_df.to_csv("rf_classification_report.csv")
logreg_report_df.to_csv("logreg_classification_report.csv")
svm_report_df.to_csv("svm_classification_report.csv")
knn_report_df.to_csv("knn_classification_report.csv")

# Print confusion matrices
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

print("Logistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, logreg_pred))

print("Support Vector Machine Confusion Matrix:")
print(confusion_matrix(y_test, svm_pred))

print("K-Nearest Neighbors Confusion Matrix:")
print(confusion_matrix(y_test, knn_pred))
