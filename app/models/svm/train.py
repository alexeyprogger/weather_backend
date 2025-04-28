import pickle
from sklearn.svm import SVC
from app.utils.data_preparation import load_and_prepare_data

training_data = load_and_prepare_data("data/Samara_train.xls")

X = [record[1:] for record in training_data]
y = [record[0] for record in training_data]

clf = SVC(kernel='rbf', C=1.0, gamma='scale')
clf.fit(X, y)

MODEL_PATH = "svm_model.pkl"
with open(MODEL_PATH, "wb") as f:
    pickle.dump(clf, f)

print(f"Модель сохранена в {MODEL_PATH}")