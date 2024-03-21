import numpy as np

def calculate_accuracy(predictions, labels):
    correct = np.sum(predictions == labels)
    accuracy = (correct / len(labels)) * 100
    return accuracy
