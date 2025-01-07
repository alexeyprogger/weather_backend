from flask import Flask, jsonify, request
from models import load_model
from weather import handle_weather_request

app = Flask(__name__)

# Загрузка нейронной сети
MODEL_PATH = "neural_network_model.pkl"
nn = load_model(MODEL_PATH)

@app.route('/predict', methods=['GET'])
def predict_frost_route():
    # Получаем параметры запроса
    api_key = request.args.get('api_key')
    city = request.args.get('city', 'Samara')
    
    if not api_key:
        return jsonify({"error": "API key is required"}), 400

    # Обработка запроса
    result = handle_weather_request(api_key, city, nn)
    
    if 'error' in result:
        return jsonify(result), 500
    else:
        return jsonify(result), 200


if __name__ == '__main__':
    app.run(debug=True)
