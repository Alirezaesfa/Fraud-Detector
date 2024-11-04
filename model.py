import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load the preprocessed data
data = pd.read_csv("C:\Users\Alireza\OneDrive\Desktop\MSc\JOB\PROJECTS\Fraud Detection AI\creditcard.csv")  # Update with the actual path if needed

# Split the dataset into features and target variable
X = data.drop('Class', axis=1)  # Replace 'Class' with your actual target column name
y = data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model's performance
print(classification_report(y_test, predictions))
