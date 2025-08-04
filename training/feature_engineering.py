# This is for the required transformation of the train and test set with feature engineering, data fixes and imputations

# feature_engineering.py
# Purpose: Feature engineering, data cleaning, imputation, and upload of processed files to S3

import os
import numpy as np
import pandas as pd
import logging
import boto3
import cloudpickle
from io import StringIO
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
import warnings
from pandas.errors import SettingWithCopyWarning
from prefect import task

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BUCKET = "mlops-zoomcamp-bike-sharing-bucket"
PREFIX = "data"
temp_pickle_path = "feature_engineered_data.pkl"
temp_cloudpickle_path = "windspeed_imputer_model.pkl"

# === Utility Functions ===
def read_csv_from_s3(bucket_name, s3_key):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket_name, Key=s3_key)
    return pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))

def upload_file_to_s3(local_path, bucket_name, s3_path):
    s3 = boto3.client("s3")
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"{local_path} does not exist")
    try:
        s3.upload_file(local_path, bucket_name, s3_path)
        print(f"✅ Uploaded {local_path} to s3://{bucket_name}/{s3_path}")
    except Exception as e:
        print(f"❌ Failed to upload {local_path}: {e}")

# Fixing the anomalous month-season mapping
def map_season(month):
    if month in ['December', 'January', 'February']:
        return 'Winter'
    elif month in ['March', 'April', 'May']:
        return 'Spring'
    elif month in ['June', 'July', 'August']:
        return 'Summer'
    else:
        return 'Fall'

# Categorize part of the day
def part_of_day(hour):
    if hour <= 6:              # 0–6     ➝ Early Morning
        return 'Early Morning'
    elif hour <= 11:           # 7–11    ➝ Morning
        return 'Morning'
    elif hour <= 15:           # 12–15   ➝ Afternoon
        return 'Afternoon'
    elif hour <= 19:           # 16–19   ➝ Evening
        return 'Evening'
    else:                      # 20–23   ➝ Night
        return 'Night'

# Temperature tolerance feeling
def label_temp(t):
    if t < 10:
        return 'brr_cold'
    elif t < 20:
        return 'cold'
    elif t < 29:
        return 'cool'
    elif t < 39:
        return 'warm'
    elif t <= 49:
        return 'hot'
    else:
        return 'boiling_hot'

@task(name="windspeed_impute_with_ML")
def windspeedImputer(df_eda_full):
    # Columns to use — categorical (string) + numerical
    wind_features = ["season", "weather", "humidity", "temp", "hour", "month", "year"]
    # Split the data
    dataWindZero = df_eda_full[df_eda_full["windspeed"] == 0].copy()
    dataWindNotZero = df_eda_full[df_eda_full["windspeed"] != 0].copy()

    # Convert feature rows to dict format for DictVectorizer
    dv = DictVectorizer(sparse=False)
    # Fit on non-zero wind data
    X_train_dict = dataWindNotZero[wind_features].to_dict(orient='records')
    X_train_vec = dv.fit_transform(X_train_dict)

    # Fit the model --> selected: RandomForestRegressor
    windspeed_imputer = RandomForestRegressor(random_state=42)
    windspeed_imputer.fit(X_train_vec, dataWindNotZero["windspeed"])

    # Transform the zero-wind data
    X_pred_dict = dataWindZero[wind_features].to_dict(orient='records')
    X_pred_vec = dv.transform(X_pred_dict)
    # Predict and fill missing values
    dataWindZero["windspeed"] = windspeed_imputer.predict(X_pred_vec)

    # Merge back and reset the indices to sort in chronologival order
    df_eda_full_imputed = pd.concat([dataWindNotZero, dataWindZero], axis=0)
    df_eda_full_imputed = df_eda_full_imputed.sort_values(by="datetime").reset_index(drop=True)

    # saving the windspeed_imputer_model for imputing windspeed of zero values records (which is in large number) for the monitoring (with Evidently) 'current dataset'
    with open(temp_cloudpickle_path, "wb") as f_out:
      cloudpickle.dump(windspeed_imputer, f_out)
    logger.info("✅ Successfully saved the windspeed imputer to be later uploaded to S3")
    
    logger.info("✅ Successfully imputed the windspeed and returning the imputed dataframe")
    return df_eda_full_imputed

@task(name="data_ingestion", retries=3, retry_delay_seconds=10)
def load_raw_data(bucket, prefix):
    try:
        df_eda_full = read_csv_from_s3(bucket, f"{prefix}/entire_data.csv")
        logger.info("✅ Loaded entire_data.csv from S3")

        df_drift_test = read_csv_from_s3(bucket, f"{prefix}/evidently_current_data.csv")
        logger.info("✅ Loaded evidently_current_data.csv from S3")

        return df_eda_full, df_drift_test

    except Exception as e:
        logger.error(f"❌ Failed to load raw data from S3: {e}")
        raise

@task(name="data_transformation")
def data_transform(df_eda_full, df_drift_test):
    logger.info("✅ Starting to feature engineer.....")
    # Getting the targeted data set
    # Converting 'datetime' column to datetime dtype from object dtype so that we can map and compare
    df_eda_full['datetime'] = pd.to_datetime(df_eda_full['datetime'])
    df_drift_test['datetime'] = pd.to_datetime(df_drift_test['datetime'])
    # removing the future data (meant for current dataset to measure the drift detection in monitoring pipeline)
    df_eda_full = df_eda_full[~df_eda_full['datetime'].isin(df_drift_test['datetime'])]

    # Feature Engineering
    # break the datetime column into meaningful temporal features which might help increase model training efficiency.
    # Extract hour
    df_eda_full['hour'] = df_eda_full['datetime'].dt.hour
    # Extract day name of week
    df_eda_full['day_name'] = df_eda_full['datetime'].dt.day_name()
    # Extract month name
    df_eda_full['month'] = df_eda_full['datetime'].dt.month_name()
    # Extract year
    df_eda_full['year'] = df_eda_full['datetime'].dt.year
    # mapping the numeric values of season and weather to their orignal names given in the official data label for clarity in EDA.
    # processing for the season column-->>Going by the official data labels------>> season - 1 = spring | 2 = summer | 3 = fall | 4 = winter
    df_eda_full["season"] = df_eda_full["season"].map({
        1: "Spring",
        2: "Summer",
        3: "Fall",
        4: "Winter"
    })
    # processing for the weather column
    df_eda_full["weather"] = df_eda_full["weather"].map({
        1: "Clear/Partly Cloudy",     # for: Clear, Few clouds, Partly cloudy, Partly cloudy
        2: "Mist/Cloudy",             # for: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
        3: "Light Snow/Rain",         # for: Light Snow, Light Rain + Thunderstorm + Scattered clouds, etc.
        4: "Heavy Precipitation"      # for: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
    })

    # Apply fix
    df_eda_full['season'] = df_eda_full['month'].apply(map_season)
    df_eda_full['part_of_day'] = df_eda_full['hour'].apply(part_of_day)
    # Segmentation of rush hour
    df_eda_full['rush_hour'] = df_eda_full['hour'].isin([8, 16, 17, 18, 19]).astype(int)
    df_eda_full['temp_tolerance_feel'] = df_eda_full['atemp'].apply(label_temp)
    # An interaction feature among temperature, feels-like-temperature and humidity df -->> insights drawn from the temp, atemp, humidity related plot above
    df_eda_full['temp_hum_factor'] = (df_eda_full['temp'] * (100 - df_eda_full['humidity'])) / df_eda_full['atemp']
    # for values with atemp=0, we need to tackle infinity values
    df_eda_full['temp_hum_factor'] = df_eda_full['temp_hum_factor'].replace([np.inf, -np.inf], np.nan)
    df_eda_full['temp_hum_factor'] = df_eda_full['temp_hum_factor'].fillna(df_eda_full['temp_hum_factor'].median())
    # Working hour feature
    df_eda_full['hour_workingday'] = df_eda_full['hour'] * df_eda_full['workingday']
    # mark bad_weather
    df_eda_full['bad_weather'] = df_eda_full['weather'].isin(["Light Snow/Rain", "Heavy Precipitation"]).astype(int)
    # avoid multicollinearity and drop atemp
    df_eda_full.drop(columns=['atemp'], inplace=True)
    
    logger.info("✅ Feature Engineered Successfully and returning the Feature Engineered dataframe")
    return df_eda_full

@task(name="upload_to_S3", retries=3, retry_delay_seconds=10)
def save_and_upload(df_eda_full):
    try:
        # Save temporarily in current directory
        pd.to_pickle(df_eda_full, temp_pickle_path)
        logger.info(f"✅ Pickle file saved temporarily at {temp_pickle_path}")

        # Upload to S3
        upload_file_to_s3(temp_pickle_path, BUCKET, f"{PREFIX}/{temp_pickle_path}")
        logger.info("✅ Pickle file (dataset) uploaded to S3 successfully")

        # Upload to S3
        upload_file_to_s3(temp_cloudpickle_path, BUCKET, f"{PREFIX}/{temp_cloudpickle_path}")
        logger.info("✅ Cloudpickle file (windspeed imputter model) uploaded to S3 successfully")
        
    except Exception as e:
        logger.error(f"❌ Error during save/upload: {e}")

    finally:
        # Clean up local file if it exists
        if os.path.exists(temp_pickle_path):
            try:
                os.remove(temp_pickle_path)
                logger.info(f"🧹 Temporary file {temp_pickle_path} removed")
            except Exception as cleanup_err:
                logger.warning(f"⚠️ Failed to delete temp file: {cleanup_err}")
        if os.path.exists(temp_cloudpickle_path):
            try:
                os.remove(temp_cloudpickle_path)
                logger.info(f"🧹 Temporary file {temp_cloudpickle_path} removed")
            except Exception as cleanup_err:
                logger.warning(f"⚠️ Failed to delete temp file: {cleanup_err}")

@task(name="feature_engineering_main")
def feature_engineering_main():
    df_eda_full, df_drift_test =  load_raw_data(BUCKET, PREFIX)
    df_eda_full = data_transform(df_eda_full, df_drift_test)
    df_imputed = windspeedImputer(df_eda_full)
    save_and_upload(df_imputed)


if __name__ == "__main__":
    feature_engineering_main()