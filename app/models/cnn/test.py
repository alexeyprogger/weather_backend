import keras
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from .utils import CLASSES, predict_image_3c


def test_cnn(image_path=None):
    loaded_model = keras.models.load_model("models/cnn_model.h5")

    prob = pd.DataFrame(
        {"prob": np.round(predict_image_3c(
        loaded_model, image_path, pixels=200, show=True
    ) * 100, 1)},
        index=CLASSES.keys(),
    ).sort_values("prob", ascending=False)

    return {
        "status": "success",
        "prediction": prob.index[0],
        "confidence": float(prob['prob'].iloc[0]),
        "probabilities": prob.to_dict()['prob']
    }

if __name__=='__main__':
    test_cnn()