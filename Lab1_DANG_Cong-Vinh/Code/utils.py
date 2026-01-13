
import numpy as np


def create_train_dataset():
    n_train = 100000
    max_train_card = 10

    ############## Task 1
    # Generate training samples with random cardinalities from 1 to 10
    # Each sample is padded with zeros to have length max_train_card
    
    X_train = np.zeros((n_train, max_train_card), dtype=np.int64)
    y_train = np.zeros(n_train, dtype=np.float32)
    
    for i in range(n_train):
        card = np.random.randint(1, max_train_card + 1)
        digits = np.random.randint(1, 11, size=card)
        X_train[i, max_train_card - card:] = digits
        y_train[i] = digits.sum()

    return X_train, y_train


def create_test_dataset():
    
    ############## Task 2
    # Generate test samples with cardinalities 5, 10, 15, ..., 100
    # 10,000 samples per cardinality
    # Return as lists of numpy arrays
    
    n_samples_per_card = 10000
    cardinalities = list(range(5, 101, 5)) 
    
    X_test = []
    y_test = []
    
    for card in cardinalities:
        X = np.random.randint(1, 11, size=(n_samples_per_card, card)).astype(np.int64)
        y = X.sum(axis=1).astype(np.float32)
        X_test.append(X)
        y_test.append(y)

    return X_test, y_test
