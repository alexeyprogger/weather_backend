import math
import numpy as np
import pandas as pd
from .model import load_model
from app.utils.data_preparation import load_and_prepare_data

MODEL_PATH = "neural_network_model.pkl"
nn = load_model(MODEL_PATH)

test_data = load_and_prepare_data("data/Samara_test.xls")

"""
    Оцениваем работу НС
    correct: переменная для хранения числа верно проклассифицированных экземпляров, total - всего экземпляров. 
    correct_label: Правильный класс (метка).
    inputs: Подготовленные входные данные.
    outputs: Получение выходных данных от НС.
    label: Индекс наибольшего значения (маркерное значение).
"""

correct = 0
total = 0
for i in test_data:
    correct_label = int(i[0])
    inputs = np.asarray(i[1:], dtype=float)
    if any(math.isnan(x) for x in inputs):
        continue
    outputs = nn.query(inputs)
    label = np.argmax(outputs)
    # print('Корректный маркер:', correct_label, 'Полученный маркер:', label, 'Данные:', i[1:])
    if label == correct_label:
        correct += 1
    total += 1
print('Верно проклассифицированных экземпляров: ', correct)
print('Всего экземпляров: ', total)
print("Эффективность = ", correct / total)
