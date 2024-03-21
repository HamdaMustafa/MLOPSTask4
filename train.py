import numpy as np
import pickle
from load_data import load_data, preprocess_data
from model import gradient_descent, sigmoid
from utils import calculate_accuracy

# Load and preprocess the dataset
X, y = load_data('C://Users//Dell//Desktop//MLOPS_Task4//gender.csv')
X_preprocessed = preprocess_data(X)

# Check if X_preprocessed is a sparse matrix and convert to a dense 2D NumPy array if necessary
if hasattr(X_preprocessed, "toarray"):  # This checks if X_preprocessed is a scipy sparse matrix
    X_preprocessed = X_preprocessed.toarray()

# Now, add the intercept term
X_preprocessed = np.c_[np.ones((X_preprocessed.shape[0], 1)), X_preprocessed]
# Adding intercept term
y = y[:, np.newaxis]

# Split the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Model parameters
learning_rate = 0.01
iterations = 1000
theta = np.zeros((X_train.shape[1], 1))

# Train the model
theta, cost_history = gradient_descent(X_train, y_train, theta, learning_rate, iterations)

# Save the model to disk
model_filename = 'gender_classification_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(theta, file)

# Predict and evaluate the model
predictions = sigmoid(X_test @ theta) >= 0.5
accuracy = calculate_accuracy(predictions, y_test)
print(f"Model accuracy: {accuracy}%")
