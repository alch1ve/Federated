import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flwr.client import ClientApp, NumPyClient
import tensorflow as tf

# Define the paths to your dataset
train_cats_path = r"C:\Users\aldri\federated\dataset\cats_and_dogs\train\cats"
train_dogs_path = r"C:\Users\aldri\federated\dataset\cats_and_dogs\train\dogs"
test_cats_path = r"C:\Users\aldri\federated\dataset\cats_and_dogs\test\cats"
test_dogs_path = r"C:\Users\aldri\federated\dataset\cats_and_dogs\test\dogs"

# Function to load images from a directory
def load_images_from_directory(directory, label, img_size=(32, 32)):
    images = []
    labels = []
    for img_name in os.listdir(directory):
        img_path = os.path.join(directory, img_name)
        img = load_img(img_path, target_size=img_size)
        img_array = img_to_array(img)
        images.append(img_array)
        labels.append(label)
    return images, labels

# Load train dataset
train_cats_images, train_cats_labels = load_images_from_directory(train_cats_path, 0)
train_dogs_images, train_dogs_labels = load_images_from_directory(train_dogs_path, 1)
x_train = np.array(train_cats_images + train_dogs_images)
y_train = np.array(train_cats_labels + train_dogs_labels)

# Load test dataset
test_cats_images, test_cats_labels = load_images_from_directory(test_cats_path, 0)
test_dogs_images, test_dogs_labels = load_images_from_directory(test_dogs_path, 1)
x_test = np.array(test_cats_images + test_dogs_images)
y_test = np.array(test_cats_labels + test_dogs_labels)

# Normalize the datasets
x_train, x_test = x_train / 255.0, x_test / 255.0

# Split train data further if needed for validation
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Partition the dataset
def partition_data(x, y, num_partitions):
    partition_size = len(x) // num_partitions
    partitions = []
    for i in range(num_partitions):
        start = i * partition_size
        end = start + partition_size
        partitions.append((x[start:end], y[start:end]))
    return partitions

num_partitions = 2
train_partitions = partition_data(x_train, y_train, num_partitions)
test_partitions = partition_data(x_test, y_test, num_partitions)

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Define Flower client
class FlowerClient(NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=10, batch_size=32)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}

def client_fn(cid: str):
    """Create and return an instance of Flower `Client`."""
    return FlowerClient().to_client()

# Parse arguments
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--partition-id",
    type=int,
    choices=[0, 1],
    default=0,
    help="Partition of the dataset (0, 1 or 2). "
)
args, _ = parser.parse_known_args()

# Load the partitioned data based on the partition id
x_train, y_train = train_partitions[args.partition_id]
x_test, y_test = test_partitions[args.partition_id]

# Define CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')  # 2 classes: cats and dogs
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)

# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client

    start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient().to_client(),
    )
