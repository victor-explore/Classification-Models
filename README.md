# Classification Models

This repository contains implementations of three different machine learning models for classifying the Kuzushiji-MNIST dataset: Logistic Regression, SVM with Gaussian Kernel, and Multi-Layer Perceptron (MLP).

## Dataset

The Kuzushiji-MNIST dataset consists of handwritten Japanese characters (kuzushiji) with:
- 10 classes
- 28x28 pixel grayscale images
- 70,000 examples (60,000 training, 10,000 test)

## Models Implemented

### 1. Logistic Regression
- Binary classifier adapted for multi-class classification
- Features:
  - Sigmoid activation function
  - Cross-entropy loss
  - L2 regularization
  - Gradient descent optimization

### 2. SVM with Gaussian Kernel
- One-vs-Rest (OvR) approach for multi-class classification
- Features:
  - Gaussian (RBF) kernel
  - Quadratic programming optimization
  - Support vector identification
  - Hyperparameters: C (regularization) and gamma (kernel coefficient)

### 3. Multi-Layer Perceptron (MLP)
- Neural network with fully connected layers
- Features:
  - Configurable hidden layers
  - Cross-entropy loss
  - L2 regularization
  - Mini-batch gradient descent
  - Customizable learning rate

## Requirements

```
numpy
matplotlib
scipy
requests
tqdm
```

## Usage

1. First, load and preprocess the data:
```python
# Load the dataset
train_images = np.load('kmnist-train-imgs.npz')['arr_0']
train_labels = np.load('kmnist-train-labels.npz')['arr_0']
test_images = np.load('kmnist-test-imgs.npz')['arr_0']
test_labels = np.load('kmnist-test-labels.npz')['arr_0']

# Preprocess
train_images = train_images.reshape((len(train_images), 28, 28, 1)) / 255.0
test_images = test_images.reshape((len(test_images), 28, 28, 1)) / 255.0
```

2. Train and evaluate the desired model (example with Logistic Regression):
```python
# Initialize model
weights = np.random.randn(input_size, output_size) * 0.01
bias = np.zeros(output_size)

# Train
for epoch in range(num_epochs):
    # Training code here...
    
# Evaluate
test_predicted_labels = np.argmax(test_predicted_probabilities, axis=1)
accuracy = np.mean(test_predicted_labels == test_labels)
```

## Model Performance

Each model can be evaluated using accuracy metrics on the test set. Performance may vary based on hyperparameter tuning and training conditions.

## Contributing

Feel free to open issues or submit pull requests with improvements.

---
Note: This is a basic implementation for educational purposes. For production use, consider using established machine learning libraries like scikit-learn or PyTorch.****
