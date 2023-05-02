import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import numpy as np
import pandas as pd
import pdb

def load_dataset(name):
    if name == "daily-climate":
        data = pd.read_csv('./datasets/daily-climate.csv')['meantemp'].to_numpy().astype(float)
    if name == "ms-stock":
        data = (pd.read_csv('./datasets/ms-stock.csv')['Open'].to_numpy()).astype(float)
    if name == "AMZN":
        allstocks = pd.read_csv('./datasets/djia.csv')
        data = (allstocks[allstocks.Name == "AMZN"]['Open'].to_numpy()).astype(float)
    if name == "GOOGL":
        allstocks = pd.read_csv('./datasets/djia.csv')
        data = (allstocks[allstocks.Name == "GOOGL"]['Open'].to_numpy()).astype(float)
    if name == "COVID-national-cases-1wk":
        cases_data = pd.read_csv('./datasets/cases.csv')
        cases_data = cases_data[(cases_data.forecaster == 'COVIDhub-4_week_ensemble')][['actual', 'target_end_date']]
        preds_data = pd.read_csv('./datasets/preds_cases.csv')
        preds_data = preds_data[(preds_data['signal'] == 'confirmed_incidence_num') & (preds_data['geo_value'] == 'us') & (preds_data['ahead'] == 1) & (preds_data.forecaster == 'COVIDhub-4_week_ensemble')][['target_end_date', 'quantile', 'value']]
        data = pd.merge(cases_data, preds_data, on='target_end_date').dropna().drop_duplicates()
        data = pd.DataFrame([data[data['quantile'] == 0.1]['actual'].to_numpy(), data[data['quantile'] == 0.1]['value'].to_numpy(), data[data['quantile'] == 0.5]['value'].to_numpy(), data[data['quantile'] == 0.9]['value'].to_numpy()], index=['actual', 'lower', 'middle', 'upper']).T
    if name == "COVID-national-cases-4wk":
        cases_data = pd.read_csv('./datasets/cases.csv')
        cases_data = cases_data[(cases_data.forecaster == 'COVIDhub-4_week_ensemble')][['actual', 'target_end_date']]
        preds_data = pd.read_csv('./datasets/preds_cases.csv')
        preds_data = preds_data[(preds_data['signal'] == 'confirmed_incidence_num') & (preds_data['geo_value'] == 'us') & (preds_data['ahead'] == 4) & (preds_data.forecaster == 'COVIDhub-4_week_ensemble')][['target_end_date', 'quantile', 'value']]
        data = pd.merge(cases_data, preds_data, on='target_end_date').dropna().drop_duplicates()
        data = pd.DataFrame([data[data['quantile'] == 0.1]['actual'].to_numpy(), data[data['quantile'] == 0.1]['value'].to_numpy(), data[data['quantile'] == 0.5]['value'].to_numpy(), data[data['quantile'] == 0.9]['value'].to_numpy()], index=['actual', 'lower', 'middle', 'upper']).T
    if name == "COVID-national-deaths-1wk":
        death_data = pd.read_csv('./datasets/deaths.csv')
        death_data = death_data[(death_data.forecaster == 'COVIDhub-ensemble')][['actual', 'target_end_date']]
        preds_data = pd.read_csv('./datasets/preds_deaths.csv')
        preds_data = preds_data[(preds_data['signal'] == 'deaths_incidence_num') & (preds_data['geo_value'] == 'us') & (preds_data['ahead'] == 1)][['target_end_date', 'quantile', 'value']]
        data = pd.merge(death_data, preds_data, on='target_end_date').dropna().drop_duplicates()
        data = pd.DataFrame([data[data['quantile'] == 0.05]['actual'].to_numpy(), data[data['quantile'] == 0.05]['value'].to_numpy(), data[data['quantile'] == 0.5]['value'].to_numpy(), data[data['quantile'] == 0.95]['value'].to_numpy()], index=['actual', 'lower', 'middle', 'upper']).T
    if name == "COVID-national-deaths-4wk":
        death_data = pd.read_csv('./datasets/deaths.csv')
        death_data = death_data[(death_data.forecaster == 'COVIDhub-ensemble')][['actual', 'target_end_date']]
        preds_data = pd.read_csv('./datasets/preds_deaths.csv')
        preds_data = preds_data[(preds_data['signal'] == 'deaths_incidence_num') & (preds_data['geo_value'] == 'us') & (preds_data['ahead'] == 4)][['target_end_date', 'quantile', 'value']]
        data = pd.merge(death_data, preds_data, on='target_end_date').dropna().drop_duplicates()
        data = pd.DataFrame([data[data['quantile'] == 0.05]['actual'].to_numpy(), data[data['quantile'] == 0.05]['value'].to_numpy(), data[data['quantile'] == 0.5]['value'].to_numpy(), data[data['quantile'] == 0.95]['value'].to_numpy()], index=['actual', 'lower', 'middle', 'upper']).T
    if name == "elec2":
        elec2_data = pd.read_csv('./datasets/elec2.csv')
        data = elec2_data['transfer'].to_numpy()
        data = data[17480+338:] # There are some constant values at the beginning of this dataset
    if name == "M4":
        data = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly_subset/train.csv")
        pdb.set_trace()
    return data
