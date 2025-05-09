import os
from keras.models import Model
from keras.layers import (Conv2D, Input, MaxPool2D, GlobalAveragePooling2D,
    Dense, Dropout, Flatten, BatchNormalization)
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.losses import CategoricalCrossentropy
from keras.regularizers import l2
from keras.optimizers import Adam

def build_model():
    model = Sequential([
    Input(shape=(200, 200, 3)),

    Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),

    Flatten(),
    Dense(units=512, activation='relu'),
    Dense(units=1, activation='sigmoid'),
])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

   

def save_model(model, path="models/cnn_model.h5"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"\nМодель сохранена в {path}\n")