import numpy as np
import pandas as pd
from .model import load_model
from app.utils.data_preparation import filter_data_by_time, decode_temp, decode_rosa, decode_rp5_cloudiness, decode_humidity

MODEL_PATH = "neural_network_model.pkl"
nn = load_model(MODEL_PATH)

test_data_list = pd.read_excel("data/Samara_test.xls")
test_data_list['Местное время в Самаре'] = pd.to_datetime(test_data_list['Местное время в Самаре'], dayfirst=True)

time_points = ['13:00', '07:00', '04:00', '01:00', '22:00']
filtered_test_data = { time: filter_data_by_time(test_data_list, time) for time in time_points }

decoded_test_data = {
    '13:00': {
        'T': decode_temp(filtered_test_data['13:00']),
        'Td': decode_rosa(filtered_test_data['13:00']),
        'N': decode_rp5_cloudiness(filtered_test_data['13:00']),
        'U': decode_humidity(filtered_test_data['13:00'])
    },
    '22:00': {
        'T': decode_temp(filtered_test_data['22:00']),
        'Td': decode_rosa(filtered_test_data['22:00']),
        'N': decode_rp5_cloudiness(filtered_test_data['22:00']),
        'U': decode_humidity(filtered_test_data['22:00'])
    },
    '07:00': filtered_test_data['07:00']['T'].to_list(),
    '04:00': filtered_test_data['04:00']['T'].to_list(),
    '01:00': filtered_test_data['01:00']['T'].to_list()
}

metkaN = [
    0 if min(decoded_test_data['01:00'][i], decoded_test_data['04:00'][i], decoded_test_data['07:00'][i]) >= 0 else 1
    for i in range(len(decoded_test_data['01:00']))
]

ready_lineN = [
    [
        metkaN[i], 
        decoded_test_data['13:00']['T'].to_list()[i],
        decoded_test_data['22:00']['T'].to_list()[i],
        decoded_test_data['13:00']['Td'].to_list()[i],
        decoded_test_data['22:00']['Td'].to_list()[i],
        decoded_test_data['13:00']['U'].to_list()[i],
        decoded_test_data['22:00']['U'].to_list()[i],
        decoded_test_data['13:00']['N'].to_list()[i],
        decoded_test_data['22:00']['N'].to_list()[i]
    ]
    for i in range(len(decoded_test_data['13:00']['T']))
]

# Журнал оценок работы нейронной сети
scorecard = []

# Прогон тестовых данных через нейронную сеть
for i in ready_lineN:
    correct_label = int(i[0])  # Правильный класс (метка)

    # Подготовка входных данных
    inputs = np.asarray(i[1:], dtype=float)

    # Получение выходных данных от нейронной сети
    outputs = nn.query(inputs)

    # Индекс наибольшего значения является маркерным значением 
    label = np.argmax(outputs)

    print('Корректный маркер:', correct_label, 'Полученный маркер:', label, i[1:])

    if label == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)

scorecard_array = np.asarray(scorecard)
print('Верно проклассифицированных экземпляров: ', scorecard_array.sum())
print('Всего экземпляров: ', scorecard_array.size)
print("Эффективность = ", scorecard_array.sum() / scorecard_array.size)
