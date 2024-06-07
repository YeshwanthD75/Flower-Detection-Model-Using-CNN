# Flower-Detection-Model-Using-CNN

Flower Classification with Convolutional Neural Networks (CNN)

This project uses a Convolutional Neural Network (CNN) to classify images of five different types of flowers. The model is trained using a dataset of flower images stored on Google Drive and is capable of predicting the type of flower from a given image.

## Project Description

The main goal of this project is to build and train a CNN model to accurately classify images of flowers into one of five categories. The project involves the following steps:

1. Loading and preprocessing the dataset.
2. Defining the CNN model architecture.
3. Training the model on the training dataset.
4. Evaluating the model on the testing dataset.
5. Making predictions on new flower images.

This project leverages a Convolutional Neural Network (CNN) to classify images of five different types of flowers: Dandelion, Rose, Sunflower, Tulip, and Daisy. The goal is to develop a model that can accurately identify the type of flower in a given image.

The project involves the following steps:

1. Data Collection and Preprocessing: Images of the five flower types are collected, resized to a uniform size, and normalized to prepare them for training.
2. Model Architecture: A CNN model is designed with multiple convolutional layers, each followed by a leaky ReLU activation function and max-pooling layers. The model ends with fully connected dense layers and a softmax activation for classification.
3. Training and Evaluation: The model is trained on a dataset split into training and testing sets. The training process involves optimizing the model's parameters to minimize classification errors. The model's performance is evaluated on the test set to ensure it generalizes well to unseen data.
4. Prediction: The trained model can be used to predict the type of flower from new images. The prediction process includes loading and preprocessing the input image, passing it through the model, and interpreting the output to identify the flower type.

   
##Key Features

1. Data Augmentation: Techniques like rotation, flipping, and brightness adjustment are used to augment the dataset, improving the model's robustness and generalization ability.
2. Custom CNN Architecture: The model employs a custom CNN architecture with leaky ReLU activations to handle the non-linearity in the data, and max-pooling layers to reduce the dimensionality while retaining essential features.
3. Real-Time Predictions: The project includes functionality to make real-time predictions on new flower images, displaying the predicted flower type along with the input image.

##Conclusion

This project demonstrates the application of CNNs for image classification tasks, specifically for classifying different types of flowers. The model achieves reasonable accuracy and can be further improved by experimenting with different model architectures, hyperparameters, and data augmentation techniques.
