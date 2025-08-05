# importing required libraries
import pandas as pd
import numpy as np
import logging
import mlflow
import boto3
from io import BytesIO
from sklearn.feature_extraction import DictVectorizer
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.metrics import root_mean_squared_error, r2_score, mean_squared_log_error
from lightgbm import LGBMRegressor
from prefect import task

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BUCKET = "mlops-zoomcamp-bike-sharing-bucket"
PREFIX = "data"
pickle_file_name = "feature_engineered_data.pkl"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("bike_demand_prediction_experiment")

# Reads a pickle (.pkl) file from an S3 bucket and loads it into a pandas DataFrame.
@task(name="read_pickle_file_from_S3", retries=3, retry_delay_seconds=10)
def read_pickle_from_s3(bucket_name, s3_key):    
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket_name, Key=s3_key)
        buffer = BytesIO(obj['Body'].read())
        df = pd.read_pickle(buffer)
        logger.info(f"✅ Successfully loaded pickle from s3://{bucket_name}/{s3_key}")
        logger.info(f"✅ Successfully loaded dataframe from s3 with  {df.shape[0]} records")
        return df
    except Exception as e:
        logger.error(f"❌ Failed to load pickle from S3: {e}")
        raise

@task(name="pre_process_data_with_Encoding")
def preprocess_for_train(df):
    # segregate categorical and continuous features
    categoricalFeatureNames = ["season","weather", "bad_weather", "holiday","workingday", "hour", "hour_workingday", "day_name","month","year", "rush_hour", "part_of_day", "temp_tolerance_feel"]
    numericalFeatureNames = ["humidity","temp", "windspeed", "temp_hum_factor"]

    # setting categorical features datatype as str so that they are recognized by DictVectorizer for encoding
    for feature in categoricalFeatureNames:
        df[feature] = df[feature].astype(str)

    df = df.set_index('datetime')
    # split this feature engineered dataset into train and test set
    df_train = df[:'2012-04-30'] # 16 months out of 20
    df_test = df['2012-05-01': '2012-08-31'] # 4 months out of 20

    X_train = df_train[categoricalFeatureNames + numericalFeatureNames]
    y_train = df_train['count'].values

    X_test = df_test[categoricalFeatureNames + numericalFeatureNames]
    y_test = df_test['count'].values

    # Encoding
    train_dict = X_train.to_dict(orient='records')
    test_dict = X_test.to_dict(orient='records')
    # Initialize and fit DictVectorizer
    dv = DictVectorizer(sparse=False)
    X_train_vec = dv.fit_transform(train_dict)
    X_test_vec = dv.transform(test_dict)

    logger.info(f"✅ Successfully pre-preocessed and split dataset for training and testing")
    return X_train_vec, y_train, X_test_vec, y_test, dv

@task(name="hyper_parameter_tuning_experiment")
def optimized_training(X_train_vec, y_train, X_val_vec, y_val, dv, num_trials):

    def objective(params):
        with mlflow.start_run():  # start mlflow experiment tracking 
            mlflow.set_tag("model", "lightgbm")
            mlflow.log_params(params) # log the list of hyperparameters 

            model = LGBMRegressor(**params,
                                    objective='poisson',
                                    random_state=42,
                                    verbose=-1
                                ) 
            model.fit(X_train_vec, np.log1p(y_train))  # log1p transform
            y_pred = model.predict(X_val_vec)

            # evaluate metrics
            rmse = root_mean_squared_error(y_val, np.exp(y_pred))
            r2 = r2_score(y_val, np.exp(y_pred))
            msle = mean_squared_log_error(y_val, np.exp(y_pred))

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("msle", msle)
            return {'loss': rmse, 'r-squared': r2, 'log-loss':msle,  'status': STATUS_OK}

    search_space = {
        'n_estimators': scope.int(hp.quniform('n_estimators', 100, 1500, 50)),  
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)), 
        'max_depth': scope.int(hp.quniform('max_depth', 3, 15, 1)), 
        'subsample': hp.uniform('subsample', 0.6, 1.0),  
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0)
         #'objective': 'poisson',
         #'random_state': 42,
         # 'verbose': -1
    }

    rstate = np.random.default_rng(42)
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate
    )
    logger.info(f"✅ Successfully completed the {num_trials} experimnent runs")

@task(name="run_track_experiment")
def run_track_experiment():
    df = read_pickle_from_s3(BUCKET, f"{PREFIX}/{pickle_file_name}")
    X_train_vec, y_train, X_val_vec, y_val, dv = preprocess_for_train(df) # as the test data is to be used for validation
    optimized_training(X_train_vec, y_train, X_val_vec, y_val, dv, 20)


if __name__ == '__main__':
    run_track_experiment()