import pandas as pd
import numpy as np
import re
from src.utils.constants import NUMERIC_FEATURES, TARGET_FEATURE
import os
import sys
import pickle
from src.logger.logging import logging
from src.exception.exception import customexception

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.info('Exception occured in save_object utils')
        raise customexception(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object utils')
        raise customexception(e,sys)

def categorize_time(time_str):
    hour, minute = map(int, time_str.split(':'))
    
    total_minutes = hour * 60 + minute
    
    if 0 <= total_minutes < 360:  # 00:00 - 05:59
        return 'Late_Night'
    elif 360 <= total_minutes < 480:  # 06:00 - 07:59
        return 'Early_Morning'
    elif 480 <= total_minutes < 720:  # 08:00 - 11:59
        return 'Morning'
    elif 720 <= total_minutes < 1020:  # 12:00 - 16:59
        return 'Afternoon'
    elif 1020 <= total_minutes < 1140:  # 17:00 - 18:59
        return 'Evening'
    elif 1140 <= total_minutes <= 1439:  # 19:00 - 23:59
        return 'Night'
    else:
        return 'Invalid'
    
def convert_to_minutes(time_str):
    match = re.match(r'(\d+)h\s*(\d+)m', time_str)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        return hours * 60 + minutes
    else:
        return None
    
def parse_stops(stop_desc):
    clean_desc = stop_desc.strip().replace('\n', '').replace('\t', '')
    if 'non-stop' in clean_desc.lower():
        return 0
    elif '1-stop' in clean_desc.lower():
        return 1
    elif '2+-stop' in clean_desc.lower():
        return 2
    
def preprocess_data(df2):
    # df2['flight_code'] = df2[['ch_code', 'num_code']].astype(str).agg('-'.join, axis=1)
    df2['date'] = pd.to_datetime(df2['date'], format='%d-%m-%Y')
    today = pd.to_datetime('10-02-2022', format='%d-%m-%Y')
    df2['days_prior_booked'] = (df2['date'] - today).dt.days
    df2['departure_time'] = df2['dep_time'].apply(categorize_time)
    df2['arrival_time'] = df2['arr_time'].apply(categorize_time)
    df2['flight_duration'] = df2['time_taken'].apply(convert_to_minutes)
    df2['number_of_stops'] = df2['stop'].apply(lambda x: parse_stops(x))
    
    df2[NUMERIC_FEATURES]=df2[NUMERIC_FEATURES].astype(float)
    
    return df2

def target_preprocess(y):
    y[TARGET_FEATURE] = y[TARGET_FEATURE].str.strip()
    y[TARGET_FEATURE] = y[TARGET_FEATURE].str.replace(',', '')
    y[TARGET_FEATURE] = pd.to_numeric(y[TARGET_FEATURE], errors='coerce')
    
    y[TARGET_FEATURE] = np.log(y[TARGET_FEATURE])
    
    return y 

def cap_outliers(df, percentile_low=2.5, percentile_high=97.5, req_columns=[]):
    # Select numeric columns
    numeric_cols = df[req_columns]
    
    # Calculate percentiles
    low_perc = numeric_cols.quantile(percentile_low / 100)
    high_perc = numeric_cols.quantile(percentile_high / 100)
    
    # Cap outliers
    df[req_columns] = numeric_cols.clip(lower=low_perc, upper=high_perc, axis=1)
    
    return df, low_perc, high_perc