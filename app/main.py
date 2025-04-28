from flask import Flask, jsonify, request
from app.models.neural_network import model
import os
from dotenv import load_dotenv
from app.weather.weather import weather_request

load_dotenv()
app = Flask(__name__)

MODEL_PATH_NN = os.getenv("MODEL_PATH_NN")
MODEL_PATH_SVM = os.getenv("MODEL_PATH_SVM")


DEFAULT_CITY = os.getenv("DEFAULT_CITY")

nn = model.load_model(MODEL_PATH_NN)
svm = model.load_model(MODEL_PATH_SVM)

MODEL_MAP = {
    'nn': nn,
    'svm': svm, 
}

def predict_frost(model_type):
    api_key = os.getenv("API_KEY")
    city = request.args.get('city', DEFAULT_CITY)

    if not api_key:
        return jsonify({"error": "API key is required"}), 400

    model_instance = MODEL_MAP[model_type]
    result = weather_request(api_key, city, model_instance, model_type)

    if 'error' in result:
        return jsonify({"error": "Are you sure this city exists?"}), 500
    else:
        return jsonify(result), 200
    
@app.route('/nn', methods=['GET'])
def predict_nn_frost_route():
    """ Возвращает результат предсказания заморозка """
    return predict_frost('nn')
    

@app.route('/svm', methods=['GET'])
def predict_svm_frost_route():
    """ Возвращает результат предсказания заморозка """
    return predict_frost('svm')


if __name__ == '__main__':
    app.run(debug=True)
