import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import numpy as np
import pandas as pd
import pdb

def load_dataset(name):
    if name == "daily-climate":
        df = pd.read_csv('./datasets/daily-climate.csv')
        df.rename({'date': 'timestamp', 'meantemp': 'y'}, axis='columns', inplace=True)
        df = df.drop("Unnamed: 0", axis='columns')
        data = df.melt(id_vars=['timestamp'], value_name='target')
        data.rename({'variable': 'item_id'}, axis='columns', inplace=True)
    if name == "ms-stock":
        df = pd.read_csv('./datasets/ms-stock.csv')
        df = df.drop(["High", "Low", "Close", "Volume"], axis='columns')
        df.rename({'Date': 'timestamp'}, axis='columns', inplace=True)
        df['item_id'] = 'y'
        data = df.melt(id_vars=['timestamp', 'item_id'], value_name='target')
        data.drop("variable", axis='columns', inplace=True)
    if name == "AMZN":
        df = pd.read_csv('./datasets/djia.csv')
        df = df.drop(["High", "Low", "Close", "Volume"], axis='columns')
        df.rename({'Date': 'timestamp', 'Name': 'item_id'}, axis='columns', inplace=True)
        df.replace({'item_id': {'AMZN': 'y'}}, inplace=True)
        data = df.melt(id_vars=['timestamp', 'item_id'], value_name='target')
        data.drop("variable", axis='columns', inplace=True)
        data = data.interpolate()
    if name == "GOOGL":
        df = pd.read_csv('./datasets/djia.csv')
        df = df.drop(["High", "Low", "Close", "Volume"], axis='columns')
        df.rename({'Date': 'timestamp', 'Name': 'item_id'}, axis='columns', inplace=True)
        df.replace({'item_id': {'GOOGL': 'y'}}, inplace=True)
        data = df.melt(id_vars=['timestamp', 'item_id'], value_name='target')
        data.drop("variable", axis='columns', inplace=True)
        data = data.interpolate()
    if name == "COVID-national-cases-1wk":
        cases_data = pd.read_csv('./datasets/cases.csv')
        cases_data = cases_data[(cases_data.forecaster == 'COVIDhub-4_week_ensemble')][['actual', 'target_end_date']]
        preds_data = pd.read_csv('./datasets/preds_cases.csv')
        preds_data = preds_data[(preds_data['signal'] == 'confirmed_incidence_num') & (preds_data['geo_value'] == 'us') & (preds_data['ahead'] == 1) & (preds_data.forecaster == 'COVIDhub-4_week_ensemble')][['target_end_date', 'quantile', 'value']]
        df = pd.merge(cases_data, preds_data, on='target_end_date').dropna().drop_duplicates()
        y = df[df['quantile'] == 0.1]['actual'].to_numpy()
        timestamp = df[df['quantile'] == 0.1]['target_end_date'].to_numpy()
        forecast = np.array([df[df['quantile'] == 0.1]['value'].to_numpy(), df[df['quantile'] == 0.5]['value'].to_numpy(), df[df['quantile'] == 0.9]['value'].to_numpy()], dtype='float').T
        data = pd.DataFrame([y, forecast, timestamp], index=['y', 'forecasts', 'timestamp']).T
        data = data.melt(id_vars=['timestamp'], value_name='target')
        data.rename({'variable': 'item_id'}, axis='columns', inplace=True)
    if name == "COVID-national-cases-4wk":
        cases_data = pd.read_csv('./datasets/cases.csv')
        cases_data = cases_data[(cases_data.forecaster == 'COVIDhub-4_week_ensemble')][['actual', 'target_end_date']]
        preds_data = pd.read_csv('./datasets/preds_cases.csv')
        preds_data = preds_data[(preds_data['signal'] == 'confirmed_incidence_num') & (preds_data['geo_value'] == 'us') & (preds_data['ahead'] == 4) & (preds_data.forecaster == 'COVIDhub-4_week_ensemble')][['target_end_date', 'quantile', 'value']]
        df = pd.merge(cases_data, preds_data, on='target_end_date').dropna().drop_duplicates()
        y = df[df['quantile'] == 0.1]['actual'].to_numpy()
        timestamp = df[df['quantile'] == 0.1]['target_end_date'].to_numpy()
        forecast = np.array([df[df['quantile'] == 0.1]['value'].to_numpy(), df[df['quantile'] == 0.5]['value'].to_numpy(), df[df['quantile'] == 0.9]['value'].to_numpy()], dtype='float').T
        data = pd.DataFrame([y, forecast, timestamp], index=['y', 'forecasts', 'timestamp']).T
        data = data.melt(id_vars=['timestamp'], value_name='target')
        data.rename({'variable': 'item_id'}, axis='columns', inplace=True)
    if name == "elec2":
        df = pd.read_csv('./datasets/elec2.csv')
        df['timestamp'] = pd.date_range(start='1996-5-7', end='1998-12-6 23:30:00', freq='30T', inclusive='both')
        df['class'] = (df['class'] == 'UP').astype(float)
        df.rename({'nswdemand': 'y'}, axis='columns', inplace=True)
        df = df[:2000]
        data = df.melt(id_vars=['timestamp'], value_name='target')
        data.rename({'variable': 'item_id'}, axis='columns', inplace=True)
        data.astype({'target': 'float64'})
    if name == "M4":
        data = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly_subset/train.csv")
    data = data.pivot(columns="item_id", index="timestamp", values="target")
    data.index = pd.to_datetime(data.index)
    return data

if __name__ == "__main__":
    # Iterate through all the datasets and attempt loading them
    #datasets = ['daily-climate', 'ms-stock', 'AMZN', 'GOOGL', 'COVID-national-cases-1wk', 'COVID-national-cases-4wk', 'elec2', 'M4']
    datasets = ['M4']
    for dataset in datasets:
        print(f"Loading {dataset} dataset")
        data = load_dataset(dataset)
        print(f"Loaded {dataset} dataset")
        print(data.columns)
