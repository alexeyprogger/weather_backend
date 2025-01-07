from data_preprocessing import filter_data_by_time, decode_temp, decode_rosa, decode_rp5_cloudiness, decode_humidity
import pandas as pd

def load_and_prepare_data(file_path):
    # Загрузка данных
    data = pd.read_excel(file_path)
    data['Местное время в Самаре'] = pd.to_datetime(data['Местное время в Самаре'], dayfirst=True)

    # Фильтрация по времени
    time_points = ['13:00', '07:00', '04:00', '01:00', '22:00']
    filtered_data = {time: filter_data_by_time(data, time) for time in time_points}

    # Декодирование данных
    decoded_data = {
        '13:00': {
            'T': decode_temp(filtered_data['13:00']),
            'Td': decode_rosa(filtered_data['13:00']),
            'N': decode_rp5_cloudiness(filtered_data['13:00']),
            'U': decode_humidity(filtered_data['13:00'])
        },
        '22:00': {
            'T': decode_temp(filtered_data['22:00']),
            'Td': decode_rosa(filtered_data['22:00']),
            'N': decode_rp5_cloudiness(filtered_data['22:00']),
            'U': decode_humidity(filtered_data['22:00'])
        },
        '07:00': filtered_data['07:00']['T'].to_list(),
        '04:00': filtered_data['04:00']['T'].to_list(),
        '01:00': filtered_data['01:00']['T'].to_list()
    }

    # Формирование меток
    labels = [
        0 if min(decoded_data['01:00'][i], decoded_data['04:00'][i], decoded_data['07:00'][i]) >= 0 else 1
        for i in range(len(decoded_data['01:00']))
    ]

    # Формирование входных данных
    training_data = [
        [
            labels[i],
            decoded_data['13:00']['T'].to_list()[i],
            decoded_data['22:00']['T'].to_list()[i],
            decoded_data['13:00']['Td'].to_list()[i],
            decoded_data['22:00']['Td'].to_list()[i],
            decoded_data['13:00']['U'].to_list()[i],
            decoded_data['22:00']['U'].to_list()[i],
            decoded_data['13:00']['N'].to_list()[i],
            decoded_data['22:00']['N'].to_list()[i]
        ]
        for i in range(len(decoded_data['13:00']['T']))
    ]

    return training_data
