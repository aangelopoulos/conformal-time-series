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
    if name == "MSFT":
        df = pd.read_csv('./datasets/djia.csv')
        df = df.drop(["High", "Low", "Close", "Volume"], axis='columns')
        df.rename({'Date': 'timestamp', 'Name': 'item_id'}, axis='columns', inplace=True)
        df.replace({'item_id': {'MSFT': 'y'}}, inplace=True)
        data = df.melt(id_vars=['timestamp', 'item_id'], value_name='target')
        data.drop("variable", axis='columns', inplace=True)
    if name == "AMZN":
        df = pd.read_csv('./datasets/djia.csv')
        df = df.drop(["High", "Low", "Close", "Volume"], axis='columns')
        df.rename({'Date': 'timestamp', 'Name': 'item_id'}, axis='columns', inplace=True)
        df.replace({'item_id': {'AMZN': 'y'}}, inplace=True)
        data = df.melt(id_vars=['timestamp', 'item_id'], value_name='target')
        data.drop("variable", axis='columns', inplace=True)
    if name == "GOOGL":
        df = pd.read_csv('./datasets/djia.csv')
        df = df.drop(["High", "Low", "Close", "Volume"], axis='columns')
        df.rename({'Date': 'timestamp', 'Name': 'item_id'}, axis='columns', inplace=True)
        df.replace({'item_id': {'GOOGL': 'y'}}, inplace=True)
        data = df.melt(id_vars=['timestamp', 'item_id'], value_name='target')
        data.drop("variable", axis='columns', inplace=True)
    if name == "COVID-deaths4wk":
        df = pd.read_pickle('./datasets/covid-ts-proc/proc_4wkdeaths.pkl')
        df.rename({'variable' : 'item_id'}, axis='columns', inplace=True)
        data = df
    if name == "tx-COVID-deaths-4wk":
        df = pd.read_pickle('./datasets/covid-ts-proc/statewide/tx_proc_4wkdeaths.pkl')
        df.rename({'variable' : 'item_id'}, axis='columns', inplace=True)
        data = df
        data = data[data.timestamp <=  np.sort(data.timestamp.unique())[105]]
    if name == "ca-COVID-deaths-4wk":
        df = pd.read_pickle('./datasets/covid-ts-proc/statewide/ca_proc_4wkdeaths.pkl')
        df.rename({'variable' : 'item_id'}, axis='columns', inplace=True)
        data = df
        data = data[data.timestamp <=  np.sort(data.timestamp.unique())[105]]
    if name == "ga-COVID-deaths-4wk":
        df = pd.read_pickle('./datasets/covid-ts-proc/statewide/ga_proc_4wkdeaths.pkl')
        df.rename({'variable' : 'item_id'}, axis='columns', inplace=True)
        data = df
        data = data[data.timestamp <=  np.sort(data.timestamp.unique())[105]]
    if name == "fl-COVID-deaths-4wk":
        df = pd.read_pickle('./datasets/covid-ts-proc/statewide/fl_proc_4wkdeaths.pkl')
        df.rename({'variable' : 'item_id'}, axis='columns', inplace=True)
        data = df
        data = data[data.timestamp <=  np.sort(data.timestamp.unique())[105]]
    if name == "ny-COVID-deaths-4wk":
        df = pd.read_pickle('./datasets/covid-ts-proc/statewide/ny_proc_4wkdeaths.pkl')
        df.rename({'variable' : 'item_id'}, axis='columns', inplace=True)
        data = df
        data = data[data.timestamp <=  np.sort(data.timestamp.unique())[105]]
    if name == "COVID-cases3wk":
        df = pd.read_pickle('./datasets/covid-ts-proc/proc_3wkcases.pkl')
        df.rename({'variable' : 'item_id'}, axis='columns', inplace=True)
        data = df
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
    data['y'] = data['y'].astype(float)
    data = data.interpolate()
    data.index = pd.to_datetime(data.index)
    return data

if __name__ == "__main__":
    # Iterate through all the datasets and attempt loading them
    datasets = ['tx-COVID-deaths-4wk', 'ca-COVID-deaths-4wk']
    for dataset in datasets:
        print(f"Loading {dataset} dataset")
        data = load_dataset(dataset)
        print(f"Loaded {dataset} dataset")
        print(data.columns)
