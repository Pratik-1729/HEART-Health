import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Loading and reading the dataset
heart = pd.read_csv("heart_health-master\\heart.csv")

# Creating a copy of the dataset to avoid affecting the original dataset
heart_df = heart.copy()

# Renaming some of the columns
print(heart_df.head())

# Model building
# Fixing our data in x and y. Here y contains target data and X contains the rest of the features.
x = heart_df.drop(columns=['target'])
y = heart_df['target']

# Splitting the dataset into training and testing using train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Feature scaling
scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_test_scaler = scaler.transform(x_test)  # Using transform instead of fit_transform for the test set

# Define the parameter grid for RandomizedSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'random_state': range(0, 5000)
}

# Initialize the RandomForestClassifier
rf = RandomForestClassifier()

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)

# Fit the model
random_search.fit(x_train_scaler, y_train)

# Get the best parameters
best_params = random_search.best_params_
print(best_params)

# Train the model with the best parameters
model = RandomForestClassifier(**best_params)
model.fit(x_train_scaler, y_train)
y_pred = model.predict(x_test_scaler)

# Printing the classification report and accuracy
print('Classification Report\n', classification_report(y_test, y_pred))
score_model = round(accuracy_score(y_pred, y_test) * 100, 2)
print('Accuracy: ' + str(score_model) + " %")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Creating a pickle file for the classifier
filename = 'heart_health-master\\heart-disease-prediction-randomforest-model.pkl'
pickle.dump(model, open(filename, 'wb'))
