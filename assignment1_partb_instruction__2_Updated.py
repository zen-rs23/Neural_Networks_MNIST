# -*- coding: utf-8 -*-
"""Assignment1_PartB_Instruction_#2 - Draft.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18_NoBiRojcEm-ZyfTXjc-VX3XXC1Wbu2

# Neural Networks with PyTorch

In this assignment, we are going to train a Neural Networks on the Japanese MNIST dataset. It is composed of 70000 images of handwritten Hiragana characters. The target variables has 10 different classes.

Each image is of dimension 28 by 28. But we will flatten them to form a dataset composed of vectors of dimension (784, 1). The training process will be similar as for a structured dataset.

<img src='https://drive.google.com/uc?id=16TqEl9ESfXYbUpVafXD6h5UpJYGKfMxE' width="500" height="200">

Your goal is to run at least 3 experiments and get a model that can achieve 80% accuracy with not much overfitting on this dataset.

Some of the code have already been defined for you. You need only to add your code in the sections specified (marked with **TODO**). Some assert statements have been added to verify the expected outputs are correct. If it does throw an error, this means your implementation is behaving as expected.

Note: You can only use fully-connected and dropout layers for this assignment. You can not convolution layers for instance

# 1. Import Required Packages

[1.1] We are going to use numpy, matplotlib and google.colab packages
"""

from google.colab import drive
import numpy as np
import matplotlib.pyplot as plt

"""# 2. Download Dataset

We will store the dataset into your personal Google Drive.

[2.1] Mount Google Drive
"""

drive.mount('/content/gdrive')

"""[2.2] Create a folder called `DL_ASG_1` on your Google Drive at the root level"""

! mkdir -p /content/gdrive/MyDrive/DL_ASG_1

"""[2.3] Navigate to this folder"""

# Commented out IPython magic to ensure Python compatibility.
# %cd '/content/gdrive/MyDrive/DL_ASG_1'

"""[2.4] Show the list of item on the folder"""

!ls

"""[2.4] Dowload the dataset files to your Google Drive if required"""

import requests
from tqdm import tqdm
import os.path

def download_file(url):
    path = url.split('/')[-1]
    if os.path.isfile(path):
        print (f"{path} already exists")
    else:
      r = requests.get(url, stream=True)
      with open(path, 'wb') as f:
          total_length = int(r.headers.get('content-length'))
          print('Downloading {} - {:.1f} MB'.format(path, (total_length / 1024000)))
          for chunk in tqdm(r.iter_content(chunk_size=1024), total=int(total_length / 1024) + 1, unit="KB"):
              if chunk:
                  f.write(chunk)

url_list = [
    'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-imgs.npz',
    'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-labels.npz',
    'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-imgs.npz',
    'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-labels.npz'
]

for url in url_list:
    download_file(url)

"""[2.5] List the content of the folder and confirm files have been dowloaded properly"""

! ls

"""# 3. Load Data

[3.1] Import the required modules from PyTorch
"""

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt

"""[3.2] **TODO** Create 2 variables called `img_height` and `img_width` that will both take the value 28"""

img_height = 28
img_width = 28

"""[3.3] Create a function that loads a .npz file using numpy and return the content of the `arr_0` key"""

def load(f):
    return np.load(f)['arr_0']

"""[3.4] **TODO** Load the 4 files saved on your Google Drive into their respective variables: x_train, y_train, x_test and y_test"""

x_train = load('/content/gdrive/MyDrive/DL_ASG_1/kmnist-train-imgs.npz')
x_test = load('/content/gdrive/MyDrive/DL_ASG_1/kmnist-test-imgs.npz')
y_train = load('/content/gdrive/MyDrive/DL_ASG_1/kmnist-train-labels.npz')
y_test = load('/content/gdrive/MyDrive/DL_ASG_1/kmnist-test-labels.npz')

"""[3.5] **TODO** Using matplotlib display the first image from the train set and its target value"""

plt.imshow(x_train[0], cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.axis('off')
plt.show()

"""# 4. Prepare Data

[4.1] **TODO** Reshape the images from the training and testing set to have the channel dimension last. The dimensions should be: (row_number, height, width, channel)
"""

x_train = x_train.reshape(-1, img_height, img_width, 1)
x_test = x_test.reshape(-1, img_height, img_width, 1)

"""[4.2] **TODO** Cast `x_train` and `x_test` into `float32` decimals"""

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

"""[4.3] **TODO** Standardise the images of the training and testing sets. Originally each image contains pixels with value ranging from 0 to 255. after standardisation, the new value range should be from 0 to 1."""

x_train = x_train/255.0
x_test = x_test/255.0

x_train = x_train.reshape(-1, 784)  # Flatten the training images
x_test = x_test.reshape(-1, 784)    # Flatten the testing images

"""[4.4] **TODO** Create a variable called `num_classes` that will take the value 10 which corresponds to the number of classes for the target variable"""

num_classes = 10

"""[4.5] **TODO** Convert the target variable for the training and testing sets to a binary class matrix of dimension (rows, num_classes).

For example:
- class 0 will become [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
- class 1 will become [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
- class 5 will become [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
- class 9 will become [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
"""

x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

data_train = TensorDataset(x_train, y_train)
data_test = TensorDataset(x_test, y_test)

"""# 5. Define Neural Networks Architecure

[5.1] Set the seed in PyTorch for reproducing results
"""

torch.manual_seed(42)

"""[5.2] **TODO** Define the architecture of your Neural Networks and save it into a variable called `model`"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = nn.Sequential(
    nn.Linear(784, 512),  # Input layer to hidden layer 1
    nn.ReLU(),            # Activation function for hidden layer 1
    nn.Linear(512, 512),  # Hidden layer 1 to hidden layer 2
    nn.ReLU(),            # Activation function for hidden layer 2
    nn.Linear(512, 10)    # Hidden layer 2 to output layer
)

"""[5.2] **TODO** Print the summary of your model"""

model.to(device)
print(model)

"""# 6. Train Neural Networks

[6.1] **TODO** Create 2 variables called `batch_size` and `epochs` that will  respectively take the values 128 and 500
"""

BATCH_SIZE = 128
epochs = 500

"""[6.2] **TODO** Compile your model with the appropriate loss function, the optimiser of your choice and the accuracy metric"""

# Criterion (Loss function)
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

dataloader_train = DataLoader(data_train, batch_size = BATCH_SIZE, shuffle = True)
dataloader_test = DataLoader(data_test, batch_size = BATCH_SIZE, shuffle=True)

len(dataloader_train)

"""[6.3] **TODO** Train your model
using the number of epochs defined. Calculate the total loss and save it to a variable called total_loss.
"""

epoch_losses = []  # List to store loss per epoch

def train_model(model, dataloader, epochs, optimizer):
    model.train()
    for i in range(epochs):
        total_loss = 0
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_loss = total_loss / len(dataloader)
        epoch_losses.append(epoch_loss)  # Save epoch loss
        print(f'Epoch {i+1}/{epochs}, Loss: {total_loss/len(dataloader)}')

"""[6.4] **TODO** Test your model.  Initiate the model.eval() along with torch.no_grad() to turn off the gradients.

"""

def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    predicted_labels = []  # To collect all predicted labels
    true_labels = []       # To collect all true labels

    with torch.no_grad():  # No gradients needed for evaluation
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(target.cpu().numpy())

    # Calculate accuracy
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy, true_labels, predicted_labels

"""**Experiment 1**"""

train_model(model, dataloader_train, epochs, optimizer)
experiment1_test_accuracy, true_labels_exp1, predicted_labels_exp1 = evaluate_model(model, dataloader_test)

experiment1_train_loss = epoch_losses.copy()

"""**Experiment 2**: With dropout and additional layers"""

epoch_losses.clear()

# Experiment 2 - With dropout and additional layers
model_exp2 = nn.Sequential(
    nn.Linear(784, 1024),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 10)
).to(device)

optimizer_exp2 = optim.Adam(model_exp2.parameters(), lr=0.001)

train_model(model_exp2, dataloader_train, epochs, optimizer_exp2)
experiment2_test_accuracy, true_labels_exp2, predicted_labels_exp2 = evaluate_model(model_exp2, dataloader_test)

experiment2_train_loss = epoch_losses.copy()

"""**Experiment 3** : Reinitialize the model for Experiment 3 using Dropout() and adjusting the Learning Rate"""

epoch_losses.clear()

# Reinitialize the model for Experiment 3
model_exp3 = nn.Sequential(
    nn.Linear(784, 512),  # Input layer to hidden layer 1
    nn.ReLU(),
    nn.Dropout(0.5),      # Adding dropout layer for regularization
    nn.Linear(512, 512),  # Hidden layer 1 to hidden layer 2
    nn.ReLU(),
    nn.Dropout(0.5),      # Adding dropout layer for regularization
    nn.Linear(512, 10)    # Hidden layer 2 to output layer
).to(device)

optimizer_exp3 = optim.Adam(model_exp3.parameters(), lr=0.0005)

train_model(model_exp3, dataloader_train, epochs, optimizer_exp3)
experiment3_test_accuracy, true_labels_exp3, predicted_labels_exp3 = evaluate_model(model_exp3, dataloader_test)

experiment3_train_loss = epoch_losses.copy()

"""# 7. Analyse Results

[7.1] **TODO** Display the performance of your model on the training and testing sets
"""

# TODO (Students need to fill this section)
#print(f"Final Test Accuracy: {accuracy:.2f}%")
print(f"Experiment 1 Test Accuracy: {experiment1_test_accuracy:.2f}%")
print(f"Experiment 2 Test Accuracy: {experiment2_test_accuracy:.2f}%")
print(f"Experiment 3 Test Accuracy: {experiment3_test_accuracy:.2f}%")

"""[7.2] **TODO** Plot the learning curve of your model"""

# TODO (Students need to fill this section)
plt.plot(experiment1_train_loss, label='Experiment 1 Loss', color='blue')
plt.plot(experiment2_train_loss, label='Experiment 2 Loss', color='green')
plt.plot(experiment3_train_loss, label='Experiment 3 Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()
plt.show()

"""[7.3] **TODO** Display the confusion matrix on the testing set predictions"""

# TODO (Students need to fill this section)
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(true_labels, predicted_labels, experiment_number):
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix - Experiment {experiment_number}')
    plt.show()

# Plotting confusion matrices for each experiment
plot_confusion_matrix(true_labels_exp1, predicted_labels_exp1, experiment_number=1)
plot_confusion_matrix(true_labels_exp2, predicted_labels_exp2, experiment_number=2)
plot_confusion_matrix(true_labels_exp3, predicted_labels_exp3, experiment_number=3)