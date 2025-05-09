from .data_preprocessing import filter_data_by_time, decode_temp, decode_rosa, decode_rp5_cloudiness, decode_humidity
import pandas as pd

def load_and_prepare_data(file_path):
    """
    Загружает и подготавливает данные для обучения.
    Параметры:
    file_path (str): Путь к файлу Excel, содержащему данные.
    Возвращает:
    list: Список списков, где каждый вложенный список содержит данные для одного образца (метки и признаки).
    """
    data = pd.read_excel(file_path)
    data['Местное время в Самаре'] = pd.to_datetime(data['Местное время в Самаре'], dayfirst=True)

    # Фильтрация по времени
    time_points = ['13:00', '07:00', '04:00', '01:00', '22:00']
    filtered_data = { time: filter_data_by_time(data, time) for time in time_points }

    # Декодирование данных
    decoded_data = {}
    for time in ['13:00', '22:00']:
        decoded_data[time] = {
            key: decoder(filtered_data[time]).to_list()
            for key, decoder in zip(['T', 'Td', 'N', 'U'], [decode_temp, decode_rosa, decode_rp5_cloudiness, decode_humidity])
        }


    for time in ['01:00', '04:00', '07:00']:
        decoded_data[time] = filtered_data[time]['T'].tolist()

    # Формирование меток
    labels = [
        0 if min(decoded_data['01:00'][i], decoded_data['04:00'][i], decoded_data['07:00'][i]) >= 0 else 1
        for i in range(len(decoded_data['01:00']))
    ]

    # Формирование входных данных
    training_data = [
        [
            labels[i],
            decoded_data['13:00']['T'][i],
            decoded_data['22:00']['T'][i],
            decoded_data['13:00']['Td'][i],
            decoded_data['22:00']['Td'][i],
            decoded_data['13:00']['U'][i],
            decoded_data['22:00']['U'][i],
            decoded_data['13:00']['N'][i],
            decoded_data['22:00']['N'][i]
        ]
        for i in range(len(decoded_data['13:00']['T']))
    ]

    return training_data
