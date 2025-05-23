import pandas as pd

def filter_data_by_time(data, time_str):
    """
    Фильтрует данные по времени.
    Параметры:
    data: Данные, содержащие столбец с временными метками.
    time_str: Время, по которому нужно отфильтровать данные.

    Возвращает отфильтрованные данные, содержащие только записи с заданным временем.
    """
    filtered_data = data[data['Местное время в Самаре'].dt.time == pd.to_datetime(time_str).time()]
    return filtered_data.copy()

def decode_rp5_cloudiness(data):
    """
    Декодирует данные облачности нотации сайта RP5 в числовой формат.

    Параметры:
    data: данные с облачностью.

    Возвращает столбец с числовыми значениями облачности.
    """
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
    """
    Декодирует данные облачности нотации сайта WeaterAPI.
    Параметры:
    data: данные с облачностью в процентах.
    """
    return 0.01 + 0.98 * (data / 100)

def decode_temp(data):
    """
    Декодирует (нормализует) температуру.

    Параметры:
    data: Столбец с температурой.
    """
    data.loc[:, 'T'] = 0.01 * data['T'] + 0.5
    return data['T']

def decode_rosa(data):
    """
    Декодирует (нормализует) точку росы.

    Параметры:
    data: Столбец с точкой росы.
    """
    data.loc[:, 'Td'] = 0.01 * data['Td'] + 0.5
    return data['Td']

def decode_humidity(data):
    """
    Декодирует (нормализует) влажность.

    Параметры:
    data: Столбец с влажностью.
    """
    data.loc[:, 'U'] = 0.01 * data['U'] - 0.01
    return data['U']
