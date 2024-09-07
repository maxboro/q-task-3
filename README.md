# Task 3
As part of this task I create interface example use case.

This module implements a digit classification system for the MNIST dataset using three different models: Convolutional Neural Network (CNN), Random Forest (RF), and a Random Model that generates random predictions. Models itself are not implemented.

## Usage
Created and tested in Python 3.9.13. To test the module with various models:

```bash
$ python interface.py
```
It will run tests using the CNN, Random Forest, and Random models, and will demonstrate error handling for invalid model names.

## Example
```bash
$ python interface.py
```
```text
Start test for cnn model
Initial data shape: (28, 28, 1)
converted to shape (28, 28, 1)
Error: ConvNNModel is to be implemented

Start test for rf model
Initial data shape: (28, 28, 1)
converted to shape (784,)
Error: RandomForestModel is to be implemented

Start test for rand model
Initial data shape: (28, 28, 1)
converted to shape (10, 10, 1)
Prediction: 7

Start test for wrong name model
Initial data shape: (28, 28, 1)
Error: wrong name model is not available, available models: cnn, rf, rand
```
