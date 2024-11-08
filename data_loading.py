import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    """
    Load, preprocess, and split the dataset for training.
    Args:
        file_path (str): Local path to the data CSV file.
    Returns:
        Tuple: Scaled training and test sets.
    """
    # Load data
    df = pd.read_csv(file_path)
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
