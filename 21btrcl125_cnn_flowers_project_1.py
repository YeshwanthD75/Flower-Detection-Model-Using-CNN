# Mount Google Drive to access the dataset
from google.colab import drive
drive.mount('/content/drive')

# Import necessary libraries
import os
import cv2
import numpy as np

# Define the directory containing the dataset
dataset_dir = "/content/drive/MyDrive/flowers"

# Initialize lists to store images and labels
X = []
y = []

# Iterate through each class folder in the dataset directory
for class_folder in os.listdir(dataset_dir):
    class_folder_path = os.path.join(dataset_dir, class_folder)

    # Iterate through each image in the class folder
    for img_file in os.listdir(class_folder_path):
        img_path = os.path.join(class_folder_path, img_file)

        # Read and resize the image
        img = cv2.imread(img_path)
        resized_img = cv2.resize(img, (100, 100))

        # Normalize the image (in this case, no additional normalization is done)
        normalized_img = resized_img

        # Append the normalized image to X
        X.append(normalized_img)

        # Append the label (class name) to y
        y.append(class_folder)

# Convert X and y to numpy arrays
X = np.array(X)
y = np.array(y)

# Display the shape of X and y
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Convert categorical labels to integer labels
import numpy as np

# Example categorical labels
categorical_labels = y

# Find unique categories
unique_categories = np.unique(categorical_labels)

# Create a dictionary mapping categories to integers
category_to_int = {category: i for i, category in enumerate(unique_categories)}

# Convert categorical labels to integer labels
integer_labels = np.array([category_to_int[category] for category in categorical_labels])

# Print categorical and integer labels
print("Categorical labels:", categorical_labels)
print("Integer labels:", integer_labels)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, integer_labels, test_size=0.2, random_state=42)

# Print the shapes of the training and testing sets
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Function to display images along with their labels
import matplotlib.pyplot as plt
import random
def display_images(images, labels, num_images=15):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        axes[i].imshow(images[i])
        axes[i].set_title("Label: " + str(labels[i]))
        axes[i].axis('off')
    plt.show()

# Display random images along with their labels
num_images_to_display = 15
random_indices = random.sample(range(len(X_train)), num_images_to_display)
random_images = [X_train[i] for i in random_indices]
random_labels = [y_train[i] for i in random_indices]

display_images(random_images, random_labels, num_images_to_display)

# Determine the number of classes
num_classes = len(np.unique(y_train))
print("Number of classes:", num_classes)

# Delete unnecessary variables to free up memory
del X, y

# Check the shape of y_train
y_train.shape

# Define the CNN model using TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Define the leaky ReLU alpha parameter
alpha = 0.1  # You can adjust this value as needed

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), input_shape=(100, 100, 3)))
model.add(tf.keras.layers.Lambda(lambda x: tf.nn.leaky_relu(x, alpha)))  # Leaky ReLU activation
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3)))
model.add(tf.keras.layers.Lambda(lambda x: tf.nn.leaky_relu(x, alpha)))  # Leaky ReLU activation
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3)))
model.add(tf.keras.layers.Lambda(lambda x: tf.nn.leaky_relu(x, alpha)))  # Leaky ReLU activation
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3)))
model.add(tf.keras.layers.Lambda(lambda x: tf.nn.leaky_relu(x, alpha)))  # Leaky ReLU activation
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3)))
model.add(tf.keras.layers.Lambda(lambda x: tf.nn.leaky_relu(x, alpha)))  # Leaky ReLU activation
model.add(layers.Flatten())
model.add(layers.Dense(64))
model.add(tf.keras.layers.Lambda(lambda x: tf.nn.leaky_relu(x, alpha)))  # Leaky ReLU activation
model.add(layers.Dense(5, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=18, validation_data=(X_test, y_test))

# Predict the class of a random image from the test set
import random
t = random.randint(1,200)
a = X_test[t:t+1]
prediction = model.predict(a)
print(np.argmax(prediction))  # Print the predicted class index
print(category_to_int)  # Print the mapping of categories to integers

# Function to find the key corresponding to a given value in a dictionary
def find_key(dictionary, target_value):
    for key, value in dictionary.items():
        if value == target_value:
            return key
    return None

# Print the predicted flower type
print(find_key(category_to_int, np.argmax(prediction)))

# Display the image
plt.imshow(X_test[t])

# Predict the classes of all test images
a = model.predict(X_test)
label_pred = []
for i in range(len(a)):
    prediction = np.argmax(a[i])
    prediction = find_key(category_to_int, prediction)
    label_pred.append(prediction)

# Convert the true labels from integers to their corresponding categories
label = []
for i in range(len(y_test)):
    label_ = find_key(category_to_int, y_test[i])
    label.append(label_)

# Calculate and print the accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(label, label_pred)
print("Accuracy:", accuracy)



# Function to load an image, preprocess it, and make a prediction
import cv2

# Load the image
image_path = "path/to/your/flower/image.jpg"  # Replace this with the path to your flower image
flower_image = cv2.imread(image_path)

# Preprocess the image
resized_image = cv2.resize(flower_image, (100, 100))  # Resize the image to match the input size of your model
normalized_image = resized_image  # You may need to perform additional normalization if your model expects it

# Expand dimensions to match the input shape expected by the model
input_image = np.expand_dims(normalized_image, axis=0)

# Use the trained model to predict the class of the flower image
predicted_class_index = np.argmax(model.predict(input_image))

# Convert the predicted class index back to the corresponding flower type
predicted_flower_type = find_key(category_to_int, predicted_class_index)

# Display the original image along with the predicted flower type
plt.imshow(cv2.cvtColor(flower_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying with matplotlib
plt.title("Predicted Flower Type: " + predicted_flower_type)
plt.axis('off')
plt.show()
