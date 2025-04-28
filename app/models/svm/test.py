import pickle
import math
from app.utils.data_preparation import load_and_prepare_data
from sklearn.metrics import f1_score

MODEL_PATH = "svm_model.pkl"
with open(MODEL_PATH, "rb") as f:
    clf = pickle.load(f)

test_data = load_and_prepare_data("data/Samara_test.xls")

y_true = []
y_pred = []

for record in test_data:
    correct_label = int(record[0])
    inputs = record[1:]

    if any(math.isnan(x) for x in inputs):
        continue

    predicted_label = clf.predict([inputs])[0]

    y_true.append(correct_label)
    y_pred.append(predicted_label)
    
score = f1_score(y_true, y_pred, average='weighted')
print('F1-метрика:', score)