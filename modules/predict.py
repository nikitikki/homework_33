import pandas as pd
import dill
import os
import json
import logging
import csv

from datetime import datetime

# path = os.environ.get('PROJECT_PATH', '/opt/airflow/plugins')
path = os.environ.get('PROJECT_PATH', '..')
path_to_models = f'{path}/data/models/cars_pipe_prediction.pkl'
path_to_test_files = f'{path}/data/test'
# files = os.listdir('/opt/airflow/plugins/data/test')
files = os.listdir(f'{path}/data/test')

def predict():
    def make_pred(file, model):
        with open(f'{path}/data/test/{file}') as fin:
            form = json.load(fin)

        df = pd.DataFrame.from_dict([form])
        y = model.predict(df)
        new_row = {
            'id': form['id'],
            'price': form['price'],
            'pred': y[0]
        }

        return new_row

    # Выгрузка модели и составление списка с предсказаниями

    with open(path_to_models, 'rb') as f:
        model = dill.load(f)

    result_rows = list()
    headers = ['id', 'price', 'pred']
    for file in files:
        new_row = make_pred(file, model)
        result_rows.append(new_row)

    # Сохранение итогового файла
    pred_filename = f'{path}/data/predictions/cars_preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'

    with open(pred_filename, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(result_rows)

    logging.info(f'Results are saved as {pred_filename}')

if __name__ == '__main__':
    predict()