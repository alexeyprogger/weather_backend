import numpy as np
from scipy.special import expit
import pickle
import os
import logging.config

logging.config.fileConfig('logger/logging_config.ini')
logger = logging.getLogger('WeatherAppLogger')

def sigmoid(x):
    """
    Функция активации: сигмоида.
    """
    return expit(x)

class NeuralNetwork:
    """
    Класс нейронной сети.

    Поля:
    inodes (int): Количество входных узлов.
    hnodes (int): Количество скрытых узлов.
    onodes (int): Количество выходных узлов.
    wih: Матрица весов между входным и скрытым слоями.
    who: Матрица весов между скрытым и выходным слоями.
    activation_function: Функция активации, используемая в сети.
    lr: Коэффициент обучения (learning rate).

    Методы:
    train(inputs_list, targets_list): Обучение нейронной сети.
    query(inputs_list): Прогнозирование выходных значений для входных данных.
    """
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        self.activation_function = sigmoid
        self.lr = learningrate

    def train(self, inputs_list, targets_list):
        """
        Обучение нейронной сети на основе предоставленных данных.
        Параметры:
        inputs_list (list или numpy.array): Входные данные для обучения.
        targets_list (list или numpy.array): Целевые значения для обучения.
        """
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
        """
        Прогнозирование выходных значений для входных данных.
        Параметры:
        inputs_list (list или numpy.array): Входные данные для прогноза.
        Возвращает прогнозируемые выходные значения.
        """
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

def save_model(nn, filename):
    """
    Функция сохранения модели
    """
    with open(filename, 'wb') as f:
        pickle.dump(nn, f)

def load_model(filename):
    """
    Функция загрузки модели
    """
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
