import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def load_data(file_path):
    data = pd.read_csv(file_path)
    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data)
    return data

def initialize_parameters():
    W1 = np.random.rand(32, 784) - 0.5
    b1 = np.random.rand(32, 1) - 0.5
    W2 = np.random.rand(16, 32) - 0.5
    b2 = np.random.rand(16, 1) - 0.5
    W3 = np.random.rand(10, 16) - 0.5
    b3 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2, W3, b3

def relu(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    exp_Z = np.exp(Z)
    return exp_Z / np.sum(exp_Z, axis=0)

def forward_propagation(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = relu(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def relu_derivative(Z):
    return Z > 0

def one_hot_encode(Y):
    one_hot_Y = np.zeros((Y.size, 10))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_propagation(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
    one_hot_Y = one_hot_encode(Y)
    dZ3 = A3 - one_hot_Y
    dW3 = 1 / m_train * dZ3.dot(A2.T)
    db3 = 1 / m_train * np.sum(dZ3, axis=1, keepdims=True)
    dZ2 = W3.T.dot(dZ3) * relu_derivative(Z2)
    dW2 = 1 / m_train * dZ2.dot(A1.T)
    db2 = 1 / m_train * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * relu_derivative(Z1)
    dW1 = 1 / m_train * dZ1.dot(X.T)
    db1 = 1 / m_train * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2, dW3, db3

def update_parameters(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1    
    W2 -= alpha * dW2  
    b2 -= alpha * db2 
    W3 -= alpha * dW3
    b3 -= alpha * db3   
    return W1, b1, W2, b2, W3, b3

def predict_classes(A3):
    return np.argmax(A3, axis=0)

def compute_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2, W3, b3 = initialize_parameters()
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_propagation(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = backward_propagation(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y)
        alpha=alpha-0.0001
        W1, b1, W2, b2, W3, b3 = update_parameters(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = predict_classes(A3)
            print(compute_accuracy(predictions, Y))
    return W1, b1, W2, b2, W3, b3

def make_predictions(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_propagation(W1, b1, W2, b2, W3, b3, X)
    predictions = predict_classes(A3)
    return predictions

def visualize_sample(index, W1, b1, W2, b2, W3, b3, X_train, Y_train):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2, W3, b3)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

data = load_data('train.csv')
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:].astype(float) / 255.

data_train = data[1000:].T
Y_train = data_train[0]
X_train = data_train[1:].astype(float) / 255.
_, m_train = X_train.shape

W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train, 0.5, 5000)
np.savez("parameters.npz", W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
visualize_sample(15, W1, b1, W2, b2, W3, b3, X_train, Y_train)
visualize_sample(1, W1, b1, W2, b2, W3, b3, X_train, Y_train)
visualize_sample(151, W1, b1, W2, b2, W3, b3, X_train, Y_train)