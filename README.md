# Приложение для предсказания заморозков

Этот проект использует нейронную сеть для предсказания вероятности заморозков на основе погодных данных, таких как температура, точка росы, влажность и облачность в определённые моменты времени в день. Погодные данные извлекаются через API и обрабатываются для предсказания.

## Установка зависимостей

Для установки всех необходимых зависимостей создайте виртуальное окружение и установите пакеты из `requirements.txt`

```bash
python -m venv venv
venv\Scripts\activate (для Windows)
pip install -r requirements.txt
```

## Функционал

Запуск команд относительно корневой директории проекта.

1. Точка входа программы (запуск сервера):

```bash
python -m app.main
```

2. Тренировка нейронной сети:

```bash
python -m app.models.neural_network.train
```

3. Прогонка нейронной сети на тестовых данных, показатели эффективности:

```bash
python -m app.models.neural_network.test
```

4. Тренировка SVM-модели:

```bash
python -m app.models.svm.train
```

5. Прогонка SVM-модели на тестовых данных, показатели эффективности:

```bash
python -m app.models.svm.test
```