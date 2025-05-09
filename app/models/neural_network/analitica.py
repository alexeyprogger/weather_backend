import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from .model import NeuralNetwork
from app.utils.data_preparation import load_and_prepare_data
from dotenv import load_dotenv

load_dotenv()

training_data = load_and_prepare_data("data/Samara_train.xls")
validation_data = load_and_prepare_data("data/Samara_test.xls")

INPUT_NODES = int(os.getenv("INPUT_NODES"))
OUTPUT_NODES = int(os.getenv("OUTPUT_NODES"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE"))
EPOCHS_COUNT = int(os.getenv("EPOCHS_COUNT"))
HIDDEN_NODES = int(os.getenv("HIDDEN_NODES"))

# Тестируемые количества скрытых нейронов
hidden_units_list = [50, 100, 200, 350, 1500, 2500]

def evaluate_model(model, test_data):
    """Оценка точности модели на тестовых данных"""
    correct = 0
    total = 0
    for record in test_data:
        correct_label = int(record[0])
        inputs = np.asarray(record[1:], dtype=float)
        if any(math.isnan(x) for x in inputs):
            continue
        outputs = model.query(inputs)
        predicted_label = np.argmax(outputs)
        if predicted_label == correct_label:
            correct += 1
        total += 1
    return correct / total

plt.figure(figsize=(15, 6))

# Кривые обучения для разных LR
plt.subplot(1, 2, 1)
lrs = [0.001, 0.01, 0.1]
for lr in lrs:
    model = NeuralNetwork(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, learningrate=lr)
    model.train(training_data, validation_data, epochs=EPOCHS_COUNT)
    plt.plot(model.val_history, label=f'LR={lr}')
plt.title('Зависимость ошибки от LR')
plt.xlabel('Эпохи')
plt.ylabel('Ошибка')
plt.legend()

# График 2: Зависимость точности от числа нейронов
plt.subplot(1, 2, 2)
accuracy_results = []
training_times = []

for hidden_units in hidden_units_list:

    model = NeuralNetwork(INPUT_NODES, hidden_units, OUTPUT_NODES, learningrate=LEARNING_RATE)
    
    model.train(training_data, validation_data, epochs=EPOCHS_COUNT)
    training_time = model.avg_epoch_time
    
    accuracy = evaluate_model(model, validation_data)
    
    accuracy_results.append(accuracy)
    training_times.append(training_time)
    
    print(f"Нейронов: {hidden_units}, Точность: {accuracy:.3f}, Время: {training_time:.1f} сек")

plt.plot(hidden_units_list, accuracy_results, 'o-')
plt.title('Зависимость точности от числа нейронов')
plt.xlabel('Число скрытых нейронов')
plt.ylabel('Точность')
plt.grid(True)

# Добавляем второй график времени обучения
ax2 = plt.gca().twinx()
ax2.plot(hidden_units_list, training_times, 'r--')
ax2.set_ylabel('Время обучения (сек)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

plt.tight_layout()
plt.savefig('hidden_units_analysis.png')
plt.close()

results_df = pd.DataFrame({
    'hidden_units': hidden_units_list,
    'accuracy': accuracy_results,
    'training_time_sec': training_times
})
results_df.to_csv('hidden_units_results.csv', index=False)

print("\nРезультаты сохранены в hidden_units_results.csv")
print(results_df)