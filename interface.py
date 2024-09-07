"""
Interface example use case.

This module implements a digit classification system for the MNIST dataset using three 
different models: Convolutional Neural Network (CNN), Random Forest (RF), and a Random 
Model that generates random predictions. 


Classes
-------
- DigitClassificationInterface(abc.ABC): An abstract base class for digit classification models.
    |- ConvNNModel: Represents a CNN classifier.
    |- RandomForestModel: Represents an RF classifier.
    |- RandomModel: Represents a random predictor
- DigitClassifier: Uses one of the models based on user input.


Functions
---------
- test(model_type: str): Tests the `DigitClassifier` with the specified model type.


Example
-------
To test the module with various models:

    $ python interface.py

It will run tests using the CNN, Random Forest, and Random models, and will demonstrate
error handling for invalid model names.
"""
from abc import ABC, abstractmethod
import numpy as np


class DigitClassificationInterface(ABC):
    """
    An abstract base class representing a prototype for digit classification models.
    """
    @staticmethod
    @abstractmethod
    def preprocess(image: np.array):
        """Preprocess the input image. This method needs to be implemented in each subclass."""

    @abstractmethod
    def predict(self, image: np.array) -> int:
        """Predict the digit from the preprocessed input image."""


class ConvNNModel(DigitClassificationInterface):
    """Convolutional Neural Network (CNN) model for digit classification."""

    @staticmethod
    def preprocess(image: np.array) -> np.array:
        """Preprocess the input image by ensuring it is in the correct format for the CNN model."""
        assert image.shape == (28, 28, 1), f'Wrong shape: {image.shape}'
        print(f'converted to shape {image.shape}')
        return image

    def predict(self, image: np.array) -> int:
        """Predict the digit using the CNN model."""
        raise NotImplementedError('ConvNNModel is to be implemented')


class RandomForestModel(DigitClassificationInterface):
    """Random Forest model for digit classification."""
    @staticmethod
    def preprocess(image: np.array) -> np.array:
        """Preprocess the input image by flattening it into a 1D array."""
        image_arr = image.reshape(-1)
        print(f'converted to shape {image_arr.shape}')
        return image_arr

    def predict(self, image: np.array) -> int:
        """Predict the digit using the Random Forest model."""
        raise NotImplementedError('RandomForestModel is to be implemented')


class RandomModel(DigitClassificationInterface):
    """Random model that predicts digits by generating random values."""
    @staticmethod
    def preprocess(image: np.array) -> np.array:
        """Preprocess the input image by extracting a 10x10 center crop."""
        crop_size = 10

        start_x = (28 - crop_size) // 2
        end_x = start_x + crop_size
        start_y = (28 - crop_size) // 2
        end_y = start_y + crop_size

        center_crop = image[start_y:end_y, start_x:end_x, :]
        print(f'converted to shape {center_crop.shape}')
        return center_crop

    def predict(self, image: np.array) -> int:
        """Predict the digit by randomly selecting a number between 0 and 9."""
        return np.random.randint(0, 10)


class DigitClassifier:
    """Performs classification using one of available models."""
    def __init__(self, model_type: str):
        """
        Initializes the DigitClassifier with the specified model.
        
        Parameters
        ----------
        model_type : str
            The type of model to use ('cnn', 'rf', 'rand').
        """
        models = {
            "cnn": ConvNNModel,
            "rf": RandomForestModel,
            "rand": RandomModel
        }
        try:
            self.model = models[model_type]()
        except KeyError as exc:
            available_models = ', '.join(models.keys())
            raise ValueError(
                f'{model_type} model is not available, available models: {available_models}'
                ) from exc
        
    def preprocess(self, image: np.array):
        """Preprocess the input image using the selected model's preprocessing method."""
        return self.model.preprocess(image)

    def predict(self, image: np.array) -> int:
        """Predict the digit using the selected model's prediction method."""
        processed_image_data = self.preprocess(image)
        prediction = self.model.predict(processed_image_data)
        return prediction


def test(model_type: str):
    """Tests the DigitClassifier with a given model type."""

    print(f"\nStart test for {model_type} model")
    image = np.random.rand(28, 28, 1)
    print(f'Initial data shape: {image.shape}')

    try:
        dc = DigitClassifier(model_type)
        prediction = dc.predict(image)
        print(f"Prediction: {prediction}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    for model_type in ['cnn', 'rf', 'rand', 'wrong name']:
        test(model_type)

