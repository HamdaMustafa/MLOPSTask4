import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, theta):
    m = len(y)
    predictions = sigmoid(X @ theta)
    cost = (-1/m) * sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return cost

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        predictions = sigmoid(X @ theta)
        theta -= (learning_rate/m) * (X.T @ (predictions - y))
        cost_history.append(cost_function(X, y, theta))

    return theta, cost_history
