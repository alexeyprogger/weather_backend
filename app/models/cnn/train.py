# train.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from .model import build_model, save_model
from .utils import load_all_images_3channel
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator




def train_and_visualize():
    x, y = load_all_images_3channel(
        classes=["sunny", "foggy"], pixels=200
    )
    x, y = np.array(x), np.array(y)

    x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=777
    )

    model = build_model()

    callbacks = [
        EarlyStopping(patience=3, min_delta=0.0001, restore_best_weights=True),
    ]

    history = model.fit(
        x_train,
        y_train,
        epochs=20,
        callbacks=callbacks,
        class_weight={0: 1., 1: 2.},
        batch_size=64,
        validation_split=0.2,
    )


    model.evaluate(x_test, y_test)

    pd.DataFrame(history.history).plot()
    plt.show()


    predictions = model.predict(x_test)
    pred_labels = (predictions > 0.5).astype(int).flatten()

    print("Predictions shape:", predictions.shape) 
    print("Pred labels shape:", pred_labels.shape)
    print("Y_test shape:", y_test.shape)

    cm = confusion_matrix(y_test, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["sunny", "foggy"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Матрица ошибок")
    plt.grid(False)
    plt.show()

    f1 = f1_score(y_test, pred_labels, average='binary')
    print(f"F1-score: {f1:.4f}")

    print(classification_report(y_test, pred_labels, target_names=["sunny", "foggy"]))


    save_model(model)


if __name__ == "__main__":
    train_and_visualize()
