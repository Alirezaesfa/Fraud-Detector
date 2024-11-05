import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load the dataset
data = pd.read_csv("creditcard.csv")

# Define features and target variable
X = data.drop('Class', axis=1)  # All columns except the target
y = data['Class']  # Target variable

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Generate classification report
report = classification_report(y_test, y_pred, output_dict=True)

# Convert report to DataFrame for saving
report_df = pd.DataFrame(report).transpose()

# Save the report to a CSV file
report_df.to_csv("classification_report.csv")

# Print confusion matrix
print(confusion_matrix(y_test, y_pred))
