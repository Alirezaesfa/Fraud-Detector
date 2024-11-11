import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout
from kerastuner import Hyperband

# Load data and handle NaNs in the target variable
data = pd.read_csv("creditcard.csv", low_memory=False)
data = data.dropna()
X = data.drop("Class", axis=1)
y = data["Class"]
X = X[~y.isna()]
y = y.dropna()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model building functions for LSTM, GRU, and DNN
def build_lstm_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=256, step=32), input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(hp.Float('dropout', 0.0, 0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_gru_model(hp):
    model = Sequential()
    model.add(GRU(units=hp.Int('units', min_value=32, max_value=256, step=32), input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(hp.Float('dropout', 0.0, 0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_dnn_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units', min_value=32, max_value=256, step=32), activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(hp.Float('dropout', 0.0, 0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Set up Keras Tuner
tuner_lstm = Hyperband(build_lstm_model, objective='val_accuracy', max_epochs=2, factor=2, directory='tuning', project_name='lstm_tuning')
tuner_gru = Hyperband(build_gru_model, objective='val_accuracy', max_epochs=2, factor=2, directory='tuning', project_name='gru_tuning')
tuner_dnn = Hyperband(build_dnn_model, objective='val_accuracy', max_epochs=2, factor=2, directory='tuning', project_name='dnn_tuning')

# Reshape for LSTM and GRU (only for these models)
X_train_lstm = np.expand_dims(X_train, -1)
X_test_lstm = np.expand_dims(X_test, -1)

# Perform hyperparameter search
tuner_lstm.search(X_train_lstm, y_train, epochs=2, validation_split=0.2)
tuner_gru.search(X_train_lstm, y_train, epochs=2, validation_split=0.2)
tuner_dnn.search(X_train, y_train, epochs=2, validation_split=0.2)

# Retrieve best models
best_lstm = tuner_lstm.get_best_models(num_models=1)[0]
best_gru = tuner_gru.get_best_models(num_models=1)[0]
best_dnn = tuner_dnn.get_best_models(num_models=1)[0]

# Evaluate models
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print(f"{model_name} Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"{model_name} Classification Report:\n", classification_report(y_test, y_pred))

# Evaluate LSTM, GRU, and DNN
evaluate_model(best_lstm, X_test_lstm, y_test, "LSTM")
evaluate_model(best_gru, X_test_lstm, y_test, "GRU")
evaluate_model(best_dnn, X_test, y_test, "DNN")
