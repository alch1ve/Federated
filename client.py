import argparse
import os
import numpy as np
from flwr.client import ClientApp, NumPyClient
import tensorflow as tf
import dataset
import model

# Define the base path to your dataset
dataset_base_path = r"C:\Users\aldri\federated\dataset\cpe_faculty"

# Load dataset
x_train, x_test, y_train, y_test = dataset.load_dataset(dataset_base_path, test_size=0.2)

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
    help="Partition of the dataset (0 or 1)."
)
args, _ = parser.parse_known_args()

# Load the partitioned data based on the partition id
x_train, y_train = train_partitions[args.partition_id]
x_test, y_test = test_partitions[args.partition_id]

# Define CNN model
num_classes = len(np.unique(y_train))  # Number of unique classes (i.e., number of persons)
input_shape = x_train[0].shape
model = model.create_model(input_shape, num_classes)

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
