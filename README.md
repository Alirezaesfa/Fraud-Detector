# Anomaly Detection for Fraudulent Transactions

## Project Overview
This project aims to develop a real-time anomaly detection system to identify fraudulent transactions using machine learning techniques. It leverages both basic and deep learning models to enhance accuracy and robustness in detecting fraud.

## Dataset
The project utilizes the Credit Card Fraud Detection dataset, which contains transactions labeled as fraudulent or non-fraudulent.

## Installation
To set up the project, clone the repository and install the required libraries:
### First Clone:
```bash
git clone https://github.com/Alirezaesfa/Fraud-Detector.git
```
### And then install dependencies:
```bash
pip install -r Requirements.txt
```

## Usage
Run the models to start processing transactions and detecting anomalies:

PS1: If you want to train your own dataset make sure you don't forget to change the names

PS2: The Target variable is a class variable so if you are using another dataset do adjustments accordingly
### To run traditional models:
```bash
python TraditionalModels.py
```
### To run neural models:
```bash
python NeuralModels.py
```

### To push all changes:
```bash
git add .
git config --global user.name "YOURGITUSERNAME"
git config --global user.email "YOUREMAIL"
git commit -m "Leave a note about what you have changed"
git push https://your_username:your_token@GIT URL TO YOUR REPO.git master


```

## Future Work
We will explore advanced deep learning models, including autoencoders and LSTMs, to further improve detection accuracy. These models will help us capture complex fraud patterns and detect anomalies more effectively. Performance comparisons with the basic models will also be included.

## Documentation
Comprehensive documentation is provided through GitHub Actions to ensure up-to-date details on all code functions, model choices, and experimental results. The documentation includes explanations of model performance, evaluation metrics, and additional insights. See the workflows directory for automatic documentation updates.
