import requests
import numpy as np
import logging
import pandas as pd
from app.utils.data_preprocessing import decode_weatherapi_cloudiness, decode_rp5_cloudiness

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
            return {
                'T13': data_13['temp_c'],             # Температура в 13:00
                'T22': data_22['temp_c'],             # Температура в 22:00
                'Td13': data_13['dewpoint_c'],        # Точка росы в 13:00
                'Td22': data_22['dewpoint_c'],        # Точка росы в 22:00
                'U13': data_13['humidity'],           # Влажность в 13:00
                'U22': data_22['humidity'],           # Влажность в 22:00
                'N13': decode_weatherapi_cloudiness(data_13['cloud']),  # Облачность в 13:00
                'N22': decode_weatherapi_cloudiness(data_22['cloud'])   # Облачность в 22:00
            }
        else:
            raise ValueError("Не удалось найти данные за 13:00 и/или 22:00.")
    else:
        raise ConnectionError(f"Ошибка при запросе к API: {response.status_code}")

def predict_frost(nn, current_day_data):
    """
    Предсказывает возможность заморозков на основе текущих погодных данных.
    Функция преобразует данные о температуре, точке росы, влажности и облачности в формат,
    подходящий для нейронной сети, и предсказывает наличие заморозков (1 - заморозки, 0 - без заморозков).
    Параметры:
    nn: Объект нейронной сети для предсказания.
    current_day_data: Данные о погоде за текущий день, включая температуру, точку росы, влажность и облачность.
    """
    inputs = np.asarray([
    0.01 * current_day_data['T13'] + 0.5,  # Температура в 13:00, нормализация
    0.01 * current_day_data['T22'] + 0.5,  # Температура в 22:00, нормализация
    0.01 * current_day_data['Td13'] + 0.5, # Точка росы в 13:00, нормализация
    0.01 * current_day_data['Td22'] + 0.5, # Точка росы в 22:00, нормализация
    0.01 * current_day_data['U13'] - 0.01, # Влажность в 13:00, нормализация
    0.01 * current_day_data['U22'] - 0.01, # Влажность в 22:00, нормализация
    decode_rp5_cloudiness(pd.DataFrame({'N': [current_day_data['N13']]})).iloc[0], # Облачность в 13:00
    decode_rp5_cloudiness(pd.DataFrame({'N': [current_day_data['N22']]})).iloc[0]  # Облачность в 22:00
], dtype=float)

    outputs = nn.query(inputs)
    predicted_label = np.argmax(outputs)
    return predicted_label

def handle_weather_request(api_key, city, nn):
    """
    Обрабатывает запрос к API погоды, получает данные и делает предсказание на основе нейронной сети.
    """
    try:
        current_day = get_weather_data(api_key, city)
        logger.info("Данные за сегодняшний день успешно получены.") 
        frost_prediction = predict_frost(nn, current_day)
        if frost_prediction == 1:
            return {"id": 1, 
                    "is_frost": True,
                    "temp_13": current_day['T13'], 
                    "temp_22": current_day['T22'],
                    "td_13": current_day['Td13'],
                    "td_22": current_day['Td22'],
                    "humidity_13": current_day['U13'],
                    "humidity_22": current_day['U22'],
                    "clouds_13":current_day['N13'],
                    "clouds_22":current_day['N22'],
                    }
        else:
            return {"id": 2, 
                    "is_frost": False,
                    "temp_13": current_day['T13'], 
                    "temp_22": current_day['T22'],
                    "td_13": current_day['Td13'],
                    "td_22": current_day['Td22'],
                    "humidity_13": current_day['U13'],
                    "humidity_22": current_day['U22'],
                    "clouds_13":current_day['N13'],
                    "clouds_22":current_day['N22'],
            }

    except Exception as e:
        logger.error(f"Ошибка: {e}")
        return {"error": str(e)}

