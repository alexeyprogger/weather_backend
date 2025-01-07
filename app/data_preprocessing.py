import pandas as pd

# Функция фильтрации данных по времени
def filter_data_by_time(data, time_str):
    filtered_data = data[data['Местное время в Самаре'].dt.time == pd.to_datetime(time_str).time()]
    return filtered_data.copy()

# Функции декодирования
def decode_rp5_cloudiness(data):
    cloud_mapping = {
        '100%.': 0.95,
        '90  или более, но не 100%': 0.9,
        '70 – 80%.': 0.8,
        '60%.': 0.6,
        '50%.': 0.5,
        '40%.': 0.4,
        '20–30%.': 0.3,
        '10%  или менее, но не 0': 0.1,
        'Облаков нет.': 0.05,
        'Небо не видно из-за тумана и/или других метеорологических явлений.': 0.01
    }
    data.loc[:, 'N'] = data['N'].replace(cloud_mapping)
    return data['N']

def decode_weatherapi_cloudiness(data):
    return 0.01 + 0.98 * (data / 100)

def decode_temp(data):
    data.loc[:, 'T'] = 0.01 * data['T'] + 0.5
    return data['T']

def decode_rosa(data):
    data.loc[:, 'Td'] = 0.01 * data['Td'] + 0.5
    return data['Td']

def decode_humidity(data):
    data.loc[:, 'U'] = 0.01 * data['U'] - 0.01
    return data['U']
