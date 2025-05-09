from .model import NeuralNetwork, save_model
from app.utils.data_preparation import load_and_prepare_data
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

INPUT_NODES = int(os.getenv("INPUT_NODES"))
HIDDEN_NODES = int(os.getenv("HIDDEN_NODES"))
OUTPUT_NODES = int(os.getenv("OUTPUT_NODES"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE"))
EPOCHS_COUNT = int(os.getenv("EPOCHS_COUNT"))

nn = NeuralNetwork(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_RATE)

training_data = load_and_prepare_data("data/Samara_train.xls")
validation_data = load_and_prepare_data("data/Samara_test.xls")

nn.train(training_data, validation_data, epochs=EPOCHS_COUNT)
nn.plot_learning_curve()

    
MODEL_PATH = "neural_network_model.pkl"
save_model(nn, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
