from .model import NeuralNetwork, save_model
from app.utils.data_preparation import load_and_prepare_data
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

INPUT_NODES = int(os.getenv("INPUT_NODES"))
HIDDEN_NODES = int(os.getenv("HIDDEN_NODES"))
OUTPUT_NODES = int(os.getenv("OUTPUT_NODES"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE"))
EPOCHS_COUNT = int(os.getenv("EPOCHS_COUNT"))

nn = NeuralNetwork(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_RATE)

file_path = "data/Samara_train.xls"
training_data = load_and_prepare_data(file_path)

for epoch in range(EPOCHS_COUNT):
    print(f"Epoch {epoch + 1}/{EPOCHS_COUNT}")
    for i, record in enumerate(training_data):
        inputs = np.asarray(record[1:], dtype=float)
        targets = np.zeros(OUTPUT_NODES) + 0.01
        targets[int(record[0])] = 0.99
        nn.train(inputs, targets)
        
    print(f"Epoch {epoch + 1} completed")
    
MODEL_PATH = "neural_network_model.pkl"
save_model(nn, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
