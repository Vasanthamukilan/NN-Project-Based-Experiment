# Project Based Experiments
### Developed By: Vasanthamukilan M
### Rgistor Number: 212222230167

## Objective :
 Build a Multilayer Perceptron (MLP) to classify handwritten digits in python
## Steps to follow:
## Dataset Acquisition:
Download the MNIST dataset. You can use libraries like TensorFlow or PyTorch to easily access the dataset.
## Data Preprocessing:
Normalize pixel values to the range [0, 1].
Flatten the 28x28 images into 1D arrays (784 elements).
## Data Splitting:

Split the dataset into training, validation, and test sets.
Model Architecture:
## Design an MLP architecture. 
You can start with a simple architecture with one input layer, one or more hidden layers, and an output layer.
Experiment with different activation functions, such as ReLU for hidden layers and softmax for the output layer.
## Compile the Model:
Choose an appropriate loss function (e.g., categorical crossentropy for multiclass classification).Select an optimizer (e.g., Adam).
Choose evaluation metrics (e.g., accuracy).
## Training:
Train the MLP using the training set.Use the validation set to monitor the model's performance and prevent overfitting.Experiment with different hyperparameters, such as the number of hidden layers, the number of neurons in each layer, learning rate, and batch size.
## Evaluation:

Evaluate the model on the test set to get a final measure of its performance.Analyze metrics like accuracy, precision, recall, and confusion matrix.
## Fine-tuning:
If the model is not performing well, experiment with different architectures, regularization techniques, or optimization algorithms to improve performance.
## Visualization:
Visualize the training/validation loss and accuracy over epochs to understand the training process. Visualize some misclassified examples to gain insights into potential improvements.

# Program:
``` py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)

# Split the dataset into training, validation, and test sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train_categorical, test_size=0.1, random_state=42)

# Design MLP architecture
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Adjust input shape
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train_split, y_train_split, epochs=10, batch_size=128, validation_data=(X_val_split, y_val_split))

# Evaluate the model on test set
test_loss, test_acc = model.evaluate(X_test, y_test_categorical)
print("Test Accuracy:", test_acc)

# Make predictions on test set
y_pred = np.argmax(model.predict(X_test), axis=-1)

# Calculate confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Load your image
img_path = '/content/seven.png'  # Replace 'path_to_your_image.png' with the path to your image file
img = Image.open(img_path)

# Preprocess the image
img_resized = img.resize((28, 28))
img_grayscale = img_resized.convert('L')
img_array = np.array(img_grayscale) / 255.0
img_input = img_array.reshape(1, 28, 28)  # Reshape the image array to match the model's input shape

# Make predictions
predictions = model.predict(img_input)
predicted_class = np.argmax(predictions)

# Display the image and predicted class
plt.imshow(img_array, cmap='gray')
plt.title(f'Predicted Class: {predicted_class}')
plt.show()
```

## Output:
# Training Loss, Validation Loss Vs Iteration Plot:
![image](https://github.com/ShakthiSundar-K/NN-Project-Based-Experiment/assets/128116143/dbf3c22b-9357-496a-86ad-dec9627f41f7)

![image](https://github.com/ShakthiSundar-K/NN-Project-Based-Experiment/assets/128116143/2e097c94-b4b8-41b4-aff3-53b73ec2cd4a)

![image](https://github.com/ShakthiSundar-K/NN-Project-Based-Experiment/assets/128116143/7a91b547-a5d6-4c59-acc0-fb495baa0135)


# Classification Report:
![image](https://github.com/ShakthiSundar-K/NN-Project-Based-Experiment/assets/128116143/fa28c490-2bf4-4a30-ba0b-3332819d3884)

# Confusion Matrix:
![image](https://github.com/ShakthiSundar-K/NN-Project-Based-Experiment/assets/128116143/5c457ac0-a27d-4753-a26d-c53de2fd8593)

# New Sample Data Prediction:
![image](https://github.com/ShakthiSundar-K/NN-Project-Based-Experiment/assets/128116143/117cf5c6-5269-4eba-a23f-a5fc8b441fb1)

## Result:
Thus, a Multilayer Perceptron (MLP) to classify handwritten digits in python is build.
