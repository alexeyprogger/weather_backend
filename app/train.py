from models import NeuralNetwork, save_model
from data_preparation import load_and_prepare_data
import numpy as np

# Настройки нейронной сети
input_nodes = 8
hidden_nodes = 350
output_nodes = 2
learning_rate = 0.1

# Создание нейронной сети
nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# Загрузка и подготовка данных
file_path = "Samara_train.xls"
training_data = load_and_prepare_data(file_path)

# Количество эпох
epochs = 8

# Обучение на нескольких эпохах
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for i, record in enumerate(training_data):
        inputs = np.asfarray(record[1:])  # Входные данные
        targets = np.zeros(output_nodes) + 0.01  # Инициализация целевых значений
        targets[int(record[0])] = 0.99  # Установка целевого значения
        nn.train(inputs, targets)  # Обучение на одной записи
        
    print(f"Epoch {epoch + 1} completed")
    
# Сохранение модели
MODEL_PATH = "neural_network_model.pkl"
save_model(nn, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
