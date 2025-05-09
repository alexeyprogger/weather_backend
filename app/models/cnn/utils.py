import os
import sys
from typing import Tuple, List

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

CLASSES = {
    "sunny": 0,
    "foggy": 1,
}
DATA_DIR = 'data/kaggle_data'


def get_class_name(class_number):
    """ Возвращает имя класса изображения по метке класса """

    for key, val in CLASSES.items():
        if val == class_number:
            return key
    raise KeyError(f"`метка класса {class_number}` не найдена")


def load_all_images(classes, pixels: 50):

    images = []
    labels = []
    i = 0

    for class_ in classes:
        class_dir = os.path.join(DATA_DIR, class_)
        assert os.path.exists(class_dir), f"Папка класса {class_} не существует"

        files = [f for f in os.listdir(class_dir) if not f.startswith("._")]

        for file in files:

            i += 1
            sys.stdout.write(f"\r Изображений загружено: {i}")
            sys.stdout.flush()
        
            img = Image.open(os.path.join(class_dir, file))

            img = img.convert("L").resize((pixels, pixels))
            img_array = np.asarray(img).flatten() / 255

            images.append(img_array)
            labels.append(CLASSES[class_])

    return images, labels


def load_all_images_3channel(classes, pixels=50):
    images = []
    labels = []
    i = 0

    for class_ in classes:
        print(class_)
        class_dir = os.path.join(DATA_DIR, class_)
        assert os.path.exists(class_dir), f"Папка класса {class_} не существует"

        files = [f for f in os.listdir(class_dir) if not f.startswith("._")]
        print('размер класса (количество изображений):', len(files))
        for file in files:
            i += 1
            sys.stdout.write(f"\r Изображений загружено: {i}")
            sys.stdout.flush()

            img = Image.open(os.path.join(class_dir, file))

            img = img.convert("RGB").resize((pixels, pixels))
            img_array = np.asarray(img, dtype=np.float32) / 255

            images.append(img_array)
            labels.append(CLASSES[class_])

    return images, labels


def predict_image(model, file, pixels=50, show=False):
    """ Предсказание изображения """

    img = Image.open(file)

    img = img.convert("L").resize((pixels, pixels))
    img_array = np.asarray(img).flatten() / 255

    pred = get_class_name(model.predict([img_array])[0])
    prob = model.predict_proba([img_array])[0]

    if show:
        ax = plt.gca()
        ax.imshow(
            img_array.reshape((pixels, pixels)), cmap=plt.get_cmap("gray")
        )
        ax.set_title(f"{prob[0]*100:0.1f}% sunny & {prob[1]*100:0.1f}% cloudy")

    return prob

def predict_image_3c(
    model, file: str, pixels: int = 50, show: bool = False
) -> str:
    """ Предсказание с 3 каналами для CNN """

    img = Image.open(file)
    img = img.convert("RGB").resize((pixels, pixels))
    img_array = np.asarray(img) / 255

    if show:
        plt.gca().imshow(img_array)

    img_array = img_array.reshape((1, pixels, pixels, 3))
    prob = model.predict(img_array, verbose=False)[0][0]

    return np.array([1 - prob, prob])