import requests
import numpy as np
import logging
import pandas as pd
from app.utils.data_preprocessing import decode_weatherapi_cloudiness, decode_rp5_cloudiness, decode_rosa, decode_temp, decode_humidity

logger = logging.getLogger("WeatherAppLogger")

def get_weather_data(api_key, city):
    """
    Получает данные о погоде по API (WeatherAPI).

    Эта функция выполняет запрос к API погоды для получения прогноза на текущий день 
    и извлекает данные о температуре, точке росы, влажности и облачности за 13:00 и 22:00.

    Параметры:
    api_key: API ключ для доступа к WeatherAPI.
    city: Название города для получения прогноза.
    """
    url = f"http://api.weatherapi.com/v1/forecast.json?key={api_key}&q={city}&days=1&aqi=no&alerts=no"
    response = requests.get(url)
    
    if response.status_code == 200:
        weather_data = response.json()
        forecast_hourly = weather_data['forecast']['forecastday'][0]['hour']
        
        # Извлечение данных за 13:00 и 22:00
        data_13 = next((hour for hour in forecast_hourly if hour['time'].endswith("13:00")), None)
        data_22 = next((hour for hour in forecast_hourly if hour['time'].endswith("22:00")), None)
        
        if data_13 and data_22:
            weather = {
                'T13': data_13['temp_c'],             # Температура в 13:00
                'T22': data_22['temp_c'],             # Температура в 22:00
                'Td13': data_13['dewpoint_c'],        # Точка росы в 13:00
                'Td22': data_22['dewpoint_c'],        # Точка росы в 22:00
                'U13': data_13['humidity'],           # Влажность в 13:00
                'U22': data_22['humidity'],           # Влажность в 22:00
                'N13': decode_weatherapi_cloudiness(data_13['cloud']),  # Облачность в 13:00
                'N22': decode_weatherapi_cloudiness(data_22['cloud'])   # Облачность в 22:00
            }
            return weather
        else:
            raise ValueError("Не удалось найти данные за 13:00 и/или 22:00.")
    else:
        raise ConnectionError(f"Ошибка при запросе к API: {response.status_code}")


def prepare_inputs(current_day_data):
    """
    Подготавливает входные данные для модели из текущих погодных данных.
    """
    df = pd.DataFrame({
        'T': [current_day_data['T13'], current_day_data['T22']],
        'Td': [current_day_data['Td13'], current_day_data['Td22']],
        'U': [current_day_data['U13'], current_day_data['U22']],
        'N': [current_day_data['N13'], current_day_data['N22']]
    })

    temps = decode_temp(df.copy())
    dew_points = decode_rosa(df.copy())
    humidity = decode_humidity(df.copy())
    clouds = decode_rp5_cloudiness(df.copy())

    inputs = np.asarray([
        temps.iloc[0], temps.iloc[1],
        dew_points.iloc[0], dew_points.iloc[1],
        humidity.iloc[0], humidity.iloc[1],
        clouds.iloc[0], clouds.iloc[1]
    ], dtype=float)

    return inputs


def predict_frost(model, current_day_data, method):
    """
    Предсказывает возможность заморозков на основе текущих погодных данных.
    Функция преобразует данные в подходящий для нейронной сети, и предсказывает наличие заморозков (1 - заморозки, 0 - без заморозков).
    """
    inputs = prepare_inputs(current_day_data)
    
    if method == "nn":
        outputs = model.query(inputs)
        return np.argmax(outputs)
    elif method == "svm":
        return model.predict([inputs])[0]
    else:
        raise ValueError("Unknown method type. Should be 'nn' or 'svm'.")

def weather_request(api_key, city, model, method):
    """
    Обрабатывает запрос к API погоды, получает данные и делает предсказание на основе переданной модели.
    """
    try:
        current_day = get_weather_data(api_key, city)
        logger.info("Данные за сегодняшний день успешно получены.") 
        frost_prediction = predict_frost(model, current_day, method)
        data = { 
        "temp_13": current_day['T13'],
        "temp_22": current_day['T22'],
        "td_13": current_day['Td13'],
        "td_22": current_day['Td22'],
        "humidity_13": current_day['U13'],
        "humidity_22": current_day['U22'],
        "clouds_13": current_day['N13'],
        "clouds_22": current_day['N22'],
        }
        data.update(
            id = 1 if frost_prediction == 1 else 2,
            is_frost = True if frost_prediction == 1 else False
        )
        return data
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        return {"error": str(e)}