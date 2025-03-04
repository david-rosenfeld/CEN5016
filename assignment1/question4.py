import numpy as np
import pandas as pd
import time
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load training data.  The shape is (4527, 5180).
X_train = pd.read_csv("nbdata/train.csv", header=None)
# Load training target data.  The shape is (4527).
y_train = pd.read_csv("nbdata/train_labels.txt", header=None).values.ravel()

# Load test data.  The shape is (1806, 5180).
X_test = pd.read_csv("nbdata/test.csv", header=None)  # Shape: (1806, 5180)
# Load test target data.  The shape is (1806).
y_test = pd.read_csv("nbdata/test_labels.txt", header=None).values.ravel()  # Shape: (1806,)

# Instantiate a Multinomial Naive Bayes Classifier
nbc = MultinomialNB()

# Record the starting time.
start_time = time.time()
# Train the classifer.
nbc.fit(X_train, y_train)
# When complete, calculate the training time.
training_time = time.time() - start_time

# Get the predictions for the training and test data.
y_train_pred = nbc.predict(X_train)
y_test_pred = nbc.predict(X_test)

# Calculate accuracies.
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print results
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")
print(f"Training Time: {training_time:.4f} seconds")

