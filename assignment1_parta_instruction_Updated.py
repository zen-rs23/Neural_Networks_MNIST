# -*- coding: utf-8 -*-
"""Assignment1_PartA_Instruction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Im1fFzR1Wq-qHN_e0jew8ZJbfZFVg3QO

# Perceptron from scratch

In this assignment, we will be reimplementing a Neural Networks from scratch.

In part A, we are going to build a simple Perceptron on a small dataset that contains only 3 features.

<img src='https://drive.google.com/uc?id=1aUtXFBMKUumwfZ-2jmR5SIvNYPaD-t2x' width="500" height="250">

Some of the code have already been defined for you. You need only to add your code in the sections specified (marked with **TODO**). Some assert statements have been added to verify the expected outputs are correct. If it does throw an error, this means your implementation is behaving as expected.

Note: You are only allowed to use Numpy and Pandas packages for the implemention of Perceptron. You can not packages such as Sklearn or Tensorflow.

# 1. Import Required Packages

[1.1] We are going to use numpy and random packages
"""

import numpy as np
import random

"""# 2. Define Dataset

[2.1] We are going to use a simple dataset containing 3 features and 7 observations. The target variable is a binary outcome (either 0 or 1)
"""

input_set = np.array([[0,1,0], [0,0,1], [1,0,0], [1,1,0], [1,1,1], [0,1,1], [0,1,0]])
labels = np.array([[1], [0], [0], [1], [1], [0], [1]])

"""# 3. Set Initial Parameters

[3.1] Let's set the seed in order to have reproducible outcomes
"""

np.random.seed(42)

"""[3.2] **TODO**: Define a function that will create a Numpy array of a given shape with random values.


For example, `initialise_array(3,1)` will return an array of dimensions (3,1)that can look like this (values may be different):


`array([[0.37454012],
       [0.95071431],
       [0.73199394]])`
"""

def initialise_array(shape):
    return np.random.rand(*shape)

"""[3.3] **TODO**: Create a Numpy array of shape (3,1) called `init_weights` filled with random values using `initialise_array()` and print them."""

init_weights = initialise_array((3,1))
print(init_weights)

"""[3.4] **TODO**: Create a Numpy array of shape (1,) called `init_bias` filled with a random value using `initialise_array()` and print it."""

init_bias = initialise_array((1,))
print(init_bias)

"""[3.5] Assert statements to check your created variables have the expected shapes"""

assert init_weights.shape == (3, 1)
assert init_bias.shape == (1,)

"""# 4. Define Linear Function
In this section we are going to implement the linear function of a neuron:

<img src='https://drive.google.com/uc?id=1vhfpGffqletFDzMIvWkCMR2jrHE5MBy5' width="500" height="300">

[4.1] **TODO**: Define a function that will perform a dot product on the provided X and weights and add the bias to it
"""

# X: input data
# weights: weights of neural networks
# bias: bias of neural network
def linear(X, weights, bias):
    return X @ weights + bias

"""[4.2] Assert statements to check your linear function is behaving as expected"""

test_weights = [[0.37454012],[0.95071431],[0.73199394]]
test_bias = [0.59865848]
assert linear(X=input_set[0], weights=test_weights, bias=test_bias)[0] == 1.54937279
assert linear(X=input_set[1], weights=test_weights, bias=test_bias)[0] == 1.3306524199999998
assert linear(X=input_set[2], weights=test_weights, bias=test_bias)[0] == 0.9731985999999999
assert linear(X=input_set[3], weights=test_weights, bias=test_bias)[0] == 1.9239129099999999
assert linear(X=input_set[4], weights=test_weights, bias=test_bias)[0] == 2.65590685
assert linear(X=input_set[5], weights=test_weights, bias=test_bias)[0] == 2.28136673
assert linear(X=input_set[6], weights=test_weights, bias=test_bias)[0] == 1.54937279

"""# 5. Activation Function

In the forward pass, an activation function is applied on the result of the linear function. We are going to implement the sigmoid function and its derivative:

<img src='https://drive.google.com/uc?id=1LK7yjCp4KBICYNvTXzILQUzQbkm7G9xC' width="200" height="100">
<img src='https://drive.google.com/uc?id=1f5jUyw0wgiVufNqveeJVZnQc6pOrDJXD' width="300" height="100">

[5.1] **TODO**: Define a function that will implement the sigmoid function
"""

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

"""[5.2] Assert statements to check your sigmoid function is behaving as expected"""

assert sigmoid(0) == 0.5
assert sigmoid(1) == 0.7310585786300049
assert sigmoid(-1) == 0.2689414213699951
assert sigmoid(9999999999999) == 1.0
assert sigmoid(-9999999999999) == 0.0

"""[5.3] **TODO**: Define a function that will implement the derivative of the sigmoid function"""

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

"""[5.2] Assert statements to check your sigmoid_derivative function is behaving as expected"""

assert sigmoid_derivative(0) == 0.25
assert sigmoid_derivative(1) == 0.19661193324148185
assert sigmoid_derivative(-1) == 0.19661193324148185
assert sigmoid_derivative(9999999999999) == 0.0
assert sigmoid_derivative(-9999999999999) == 0.0

"""# 6. Forward Pass

Now we have everything we need to implement the forward propagation

[6.1] **TODO**: Define a function that will implement the forward pass (apply linear function on the input followed by the sigmoid activation function)
"""

def forward(X, weights, bias):
    return sigmoid(linear(X, weights, bias))

"""[6.2] Assert statements to check your forward function is behaving as expected"""

assert forward(X=input_set[0], weights=test_weights, bias=test_bias)[0] == 0.8248231247647452
assert forward(X=input_set[1], weights=test_weights, bias=test_bias)[0] == 0.7909485322272701
assert forward(X=input_set[2], weights=test_weights, bias=test_bias)[0] == 0.7257565873271445
assert forward(X=input_set[3], weights=test_weights, bias=test_bias)[0] == 0.8725741389540382
assert forward(X=input_set[4], weights=test_weights, bias=test_bias)[0] == 0.9343741240208852
assert forward(X=input_set[5], weights=test_weights, bias=test_bias)[0] == 0.9073220375080315
assert forward(X=input_set[6], weights=test_weights, bias=test_bias)[0] == 0.8248231247647452

"""# 7. Calculate Error

After the forward pass, the Neural Networks will calculate the error between its predictions (output of forward pass) and the actual targets.

[7.1] **TODO**: Define a function that will implement the error calculation (difference between predictions and actual targets)
"""

def calculate_error(actual, pred):
    return pred - actual

"""[7.2] Assert statements to check your calculate_error function is behaving as expected"""

test_actual = np.array([0,0,0,1,1,1])
assert calculate_error(actual=test_actual, pred=[0,0,0,1,1,1]).sum() == 0
assert calculate_error(actual=test_actual, pred=[0,0,0,1,1,0]).sum() == -1
assert calculate_error(actual=test_actual, pred=[0,0,0,0,0,0]).sum() == -3

"""# 8. Calculate Gradients
Once the error has been calculated, a Neural Networks will use this information to update its weights accordingly.

[8.1] Let's creata function that calculate the gradients using the sigmoid derivative function and applying the chain rule.
"""

def calculate_gradients(pred, error, input):
  dpred = sigmoid_derivative(pred)
  z_del = error * dpred
  gradients = np.dot(input.T, z_del)
  return gradients, z_del

"""# 9. Training

Now that we built all the components of a Neural Networks, we can finally train it on our dataset.

[9.1] Create 2 variables called `weights` and `bias` that will respectively take the value of `init_weights` and `init_bias`
"""

weights = init_weights
bias = init_bias

"""[9.2] Create a variable called `lr` that will be used as the learning rate for updating the weights"""

lr = 0.5

"""[9.3] Create a variable called `epochs` with the value 10000. This will the number of times the Neural Networks will process the entire dataset and update its weights"""

epochs = 10000

"""[9.4] Create a for loop that will perform the training of our Neural Networks"""

for epoch in range(epochs):
    inputs = input_set

    # Forward Propagation
    z = forward(X=inputs, weights=weights, bias=bias)

    # Error
    error = calculate_error(actual=labels, pred=z)

    # Back Propagation
    gradients, z_del = calculate_gradients(pred=z, error=error, input=input_set)

    # Update parameters
    weights = weights - lr * gradients
    for num in z_del:
        bias = bias - lr * num

"""[9.5] **TODO** Print the final values of `weights` and `bias`"""

# TODO (Students need to fill this section)
print(weights)
print(bias)

"""# 10. Compare before and after training

Let's compare the predictions of our Neural Networks before (using `init_weights` and `init_bias`) and after the training (using `weights` and `bias`)

[10.1] Create a function to display the values of a single observation from the dataset (using its index), the error and the actual target and prediction
"""

def compare_pred(weights, bias, index, X, y):
    pred = forward(X=X[index], weights=weights, bias=bias)
    actual = y[index]
    error = calculate_error(actual, pred)
    print(f"{X[index]} - Error {error} - Actual: {actual} - Pred: {pred}")

"""[10.2] Compare the results on the first observation (index 0)"""

compare_pred(weights=init_weights, bias=init_bias, index=0, X=input_set, y=labels)
compare_pred(weights=weights, bias=bias, index=0, X=input_set, y=labels)

"""[10.3] Compare the results on the second observation (index 1)"""

compare_pred(weights=init_weights, bias=init_bias, index=1, X=input_set, y=labels)
compare_pred(weights=weights, bias=bias, index=1, X=input_set, y=labels)

"""[10.4] Compare the results on the third observation (index 2)"""

compare_pred(weights=init_weights, bias=init_bias, index=2, X=input_set, y=labels)
compare_pred(weights=weights, bias=bias, index=2, X=input_set, y=labels)

"""[10.5] Compare the results on the forth observation (index 3)"""

compare_pred(weights=init_weights, bias=init_bias, index=3, X=input_set, y=labels)
compare_pred(weights=weights, bias=bias, index=3, X=input_set, y=labels)

"""[10.6] Compare the results on the fifth observation (index 4)"""

compare_pred(weights=init_weights, bias=init_bias, index=4, X=input_set, y=labels)
compare_pred(weights=weights, bias=bias, index=4, X=input_set, y=labels)

"""[10.7] Compare the results on the sixth observation (index 5)"""

compare_pred(weights=init_weights, bias=init_bias, index=5, X=input_set, y=labels)
compare_pred(weights=weights, bias=bias, index=5, X=input_set, y=labels)

"""[10.8] Compare the results on the sixth observation (index 5)"""

compare_pred(weights=init_weights, bias=init_bias, index=6, X=input_set, y=labels)
compare_pred(weights=weights, bias=bias, index=6, X=input_set, y=labels)

"""We can see after 10000 epochs, our Neural Networks is performing extremely well on our dataset. It has found pretty good values for the weights and bias to make accurate prediction.

Please submit this notebook into Canvas. Name it following this rule: *assignment1-partA-\<student_id\>.ipynb*
"""