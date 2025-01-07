import numpy as np
from scipy.special import expit
import pickle
import os
import logging.config

logging.config.fileConfig('logger/logging_config.ini')
logger = logging.getLogger('WeatherAppLogger')

def sigmoid(x):
    return expit(x)

# Класс нейронной сети
class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        self.activation_function = sigmoid
        self.lr = learningrate

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # Скрытый слой
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        # Выходной слой
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # Ошибки
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        # Обновление весов
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                     np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                     np.transpose(inputs))

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

# Функции для сохранения и загрузки модели
def save_model(nn, filename):
    with open(filename, 'wb') as f:
        pickle.dump(nn, f)

def load_model(filename):
    try:
        if not os.path.isfile(filename):
                raise FileNotFoundError(f"Файл {filename} не найден.")
        with open(filename, 'rb') as f:
            logger.info(f"Загрузка модели из файла: {filename}")
            return pickle.load(f)
    except FileNotFoundError as e:
        logger.error(f"Ошибка: {e}. Скорее всего, Вы не выполнили предварительно тренировку нейронной сети.")
        return None
    except Exception as e:
        logger.error(f"Неизвестная ошибка: {e}")
        return None
