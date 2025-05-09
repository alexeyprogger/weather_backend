import time
import numpy as np
from scipy.special import expit
import pickle
import os
import math
import logging.config
import matplotlib.pyplot as plt

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
        self.train_history = []
        self.val_history = []
        self.avg_epoch_time = 0.0

    def _forward(self, inputs: np.ndarray):
        """Прямой проход с возвратом скрытого и выходного слоев"""
        hidden = self.activation_function(np.dot(self.wih, inputs))
        outputs = self.activation_function(np.dot(self.who, hidden))
        return hidden, outputs

    def train(self, train_data, val_data, epochs, patience=2, min_delta=0.001):
        """ Обучение нейронной сети на основе предоставленных данных """
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_weights = (self.wih.copy(), self.who.copy())
        total_time = 0.0

        for epoch in range(epochs):

            train_loss = 0.0
            epoch_start = time.time()

            for record in train_data:
                if np.isnan(record).any():
                    continue
                inputs = np.array(np.asarray(record[1:], dtype=float), ndmin=2).T
                targets_train = np.zeros(self.onodes) + 0.01
                targets_train[int(record[0])] = 0.99
                targets_train = np.array(targets_train, ndmin=2).T

                training_hidden, training_outputs = self._forward(inputs)
                error = targets_train - training_outputs
                train_loss += np.mean(error**2)
            
                # Back propagation
                hidden_error = np.dot(self.who.T, error)
                self.who += self.lr * np.dot(error * training_outputs * (1 - training_outputs), training_hidden.T)
                self.wih += self.lr * np.dot(hidden_error * training_hidden * (1 - training_hidden), inputs.T)

            val_loss = 0.0
            for record in val_data:
                if any(math.isnan(x) for x in record):
                    continue
                inputs = np.array(np.asarray(record[1:], dtype=float), ndmin=2).T
                targets_val = np.zeros(self.onodes) + 0.01
                targets_val[int(record[0])] = 0.99
                targets_val = np.array(targets_val, ndmin=2).T
            
                _, val_outputs = self._forward(inputs)
                val_err = targets_val - val_outputs
                val_loss += np.mean(val_err**2)
            
            avg_train_loss = train_loss / len(train_data)
            avg_val_loss = val_loss / len(val_data)
            self.train_history.append(avg_train_loss)
            self.val_history.append(avg_val_loss)

            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_weights = (self.wih.copy(), self.who.copy())
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                logger.info(f"Ранняя остановка на эпохе {epoch+1} (нет улучшения > {min_delta} для {patience} эпох подряд)")
                self.wih, self.who = best_weights
                break
            
            epoch_time = time.time() - epoch_start
            total_time += epoch_time
            self.avg_epoch_time = total_time / (epoch + 1)  

            logger.info(f"Epoch {epoch+1}/{epochs} | Train: {round(avg_train_loss, 4)} | Val: {round(avg_val_loss, 4)} | Best: {round(best_val_loss, 4)}")


    def plot_learning_curve(self):
        """График обучения с валидацией"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_history, label='Training Loss', color='blue', marker='o')
        plt.plot(self.val_history, label='Validation Loss', color='red', marker='x')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig('learning_curve.png')
        plt.close()

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
