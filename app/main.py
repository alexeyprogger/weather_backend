from flask import Flask, jsonify, request
from app.models.neural_network import model
import os
from dotenv import load_dotenv
from app.weather.weather import weather_request
from app.models.cnn.test import test_cnn
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model as load_keras_model

load_dotenv()
app = Flask(__name__)

MODEL_PATH_NN = os.getenv("MODEL_PATH_NN")
MODEL_PATH_SVM = os.getenv("MODEL_PATH_SVM")
MODEL_PATH_CNN = os.getenv("MODEL_PATH_CNN")

UPLOAD_FOLDER = 'tmp/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

DEFAULT_CITY = os.getenv("DEFAULT_CITY")

nn = model.load_model(MODEL_PATH_NN)
svm = model.load_model(MODEL_PATH_SVM)
cnn = load_keras_model(MODEL_PATH_CNN)

MODEL_MAP = {
    'nn': nn,
    'svm': svm,
    'cnn': cnn,
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

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/nn', methods=['GET'])
def predict_nn_frost_route():
    """ Возвращает результат предсказания заморозка """
    return predict_frost('nn')
    

@app.route('/svm', methods=['GET'])
def predict_svm_frost_route():
    """ Возвращает результат предсказания заморозка """
    return predict_frost('svm')

@app.route('/cnn', methods=['POST'])
def predict_image():
    """ Обрабатывает предсказание по изображения (солнце или туман) """
    if 'file' not in request.files:
        return jsonify({"error": "Файл не найден"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Файл не найден"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.normpath(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)
        
        try:
            result = test_cnn(image_path=filepath)
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    else:
        return jsonify({"error": "Допустимые расширения файла: png, jpg, jpeg"}), 400

if __name__ == '__main__':
    app.run(debug=True)
