from flask import Flask, jsonify, request
from app.models import model
import os
from dotenv import load_dotenv
from app.weather.weather import handle_weather_request

load_dotenv()
app = Flask(__name__)

MODEL_PATH = os.getenv("MODEL_PATH")
DEFAULT_CITY = os.getenv("DEFAULT_CITY")
nn = model.load_model(MODEL_PATH)


@app.route('/predict', methods=['GET'])
def predict_frost_route():
    """
    Обрабатывает GET запросы по маршруту '/predict' и предсказывает вероятность заморозков
    на основе данных о погоде для указанного города.
    """
    api_key = os.getenv("API_KEY")
    city = request.args.get('city', DEFAULT_CITY)
    
    if not api_key:
        return jsonify({"error": "API key is required"}), 400

    result = handle_weather_request(api_key, city, nn)
    
    if 'error' in result:
        return jsonify({"error": "Are you sure this city exists?"}), 500
    else:
        return jsonify(result), 200


if __name__ == '__main__':
    app.run(debug=True)
