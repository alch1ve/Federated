import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Define the base path to your dataset
dataset_base_path = r"C:\Users\aldri\federated\cpe_faculty"

# Function to load images from a directory
def load_images_from_directory(directory, label, img_size=(32, 32)):
    images = []
    labels = []
    for img_name in os.listdir(directory):
        if img_name.endswith('.jpg'):
            img_path = os.path.join(directory, img_name)
            img = load_img(img_path, target_size=img_size)
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(label)
    return images, labels

# Function to load dataset
def load_dataset(base_path, test_size=0.2):
    classes = os.listdir(base_path)
    x_train, x_test, y_train, y_test = [], [], [], []
    for idx, class_name in enumerate(classes):
        class_path = os.path.join(base_path, class_name)
        class_images, class_labels = load_images_from_directory(class_path, idx)
        # Split images and labels into train and test sets
        x_train_class, x_test_class, y_train_class, y_test_class = train_test_split(class_images, class_labels, test_size=test_size, random_state=42)
        x_train.extend(x_train_class)
        x_test.extend(x_test_class)
        y_train.extend(y_train_class)
        y_test.extend(y_test_class)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

# Load dataset
x_train, x_test, y_train, y_test = load_dataset(dataset_base_path, test_size=0.2)

# Normalize the datasets
x_train = x_train / 255.0
x_test = x_test / 255.0
