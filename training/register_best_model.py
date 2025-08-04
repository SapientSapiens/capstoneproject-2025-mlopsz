import os
import pickle
import cloudpickle
import mlflow
import logging
import numpy as np
import pandas as pd
import boto3
from io import BytesIO
from sklearn.feature_extraction import DictVectorizer
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from lightgbm import LGBMRegressor
from sklearn.metrics import root_mean_squared_error, r2_score, mean_squared_log_error
from mlflow.models.signature import infer_signature
from prefect import task

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BUCKET = "mlops-zoomcamp-bike-sharing-bucket"
PREFIX = "data"
pickle_file_name = "feature_engineered_data.pkl"
temp_cloudpickle_filename = "dv.pkl"

OPTIMIZATION_EXPERIMENT = "bike_demand_prediction_experiment"
CURRENT_EXPERIMENT_NAME = "bike_demand_prediction_best_model"
PARAMS = ['n_estimators', 'learning_rate', 'max_depth', 'subsample', 'colsample_bytree']

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(CURRENT_EXPERIMENT_NAME)
client = MlflowClient()

@task(name="read_pickle_file_from_S3", retries=3, retry_delay_seconds=10)
def read_pickle_from_s3(bucket_name, s3_key):    
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket_name, Key=s3_key)
        buffer = BytesIO(obj['Body'].read())
        df = pd.read_pickle(buffer)
        logger.info(f"✅ Successfully loaded pickle from s3://{bucket_name}/{s3_key}")
        logger.info(f"✅ Successfully loaded dataframe from s3 with {df.shape[0]} records")
        return df
    except Exception as e:
        logger.error(f"❌ Failed to load pickle from S3: {e}")
        raise

@task(name="encoding_data_for_ML")
def preprocess_for_train(df):
    categoricalFeatureNames = [
        "season", "weather", "bad_weather", "holiday", "workingday",
        "hour", "hour_workingday", "day_name", "month", "year",
        "rush_hour", "part_of_day", "temp_tolerance_feel"
    ]
    numericalFeatureNames = ["humidity", "temp", "windspeed", "temp_hum_factor"]

    for feature in categoricalFeatureNames:
        df[feature] = df[feature].astype(str)

    df = df.set_index('datetime')
    df_train = df[:'2012-04-30']
    df_test = df['2012-05-01': '2012-08-31']

    X_train = df_train[categoricalFeatureNames + numericalFeatureNames]
    y_train = df_train['count'].values

    X_test = df_test[categoricalFeatureNames + numericalFeatureNames]
    y_test = df_test['count'].values

    dv = DictVectorizer(sparse=False)
    X_train_vec = dv.fit_transform(X_train.to_dict(orient='records'))
    X_test_vec = dv.transform(X_test.to_dict(orient='records'))
    logger.info("✅ Successfully pre-processed and split dataset")

    with open(temp_cloudpickle_filename, "wb") as f_out:
        cloudpickle.dump(dv, f_out)
    logger.info("✅ Successfully serialized DictVectorizer")

    return X_train_vec, y_train, X_test_vec, y_test, dv

def train_and_log_model(X_train_vec, y_train, X_val_vec, y_val, params):
    with mlflow.start_run() as run:
        new_params = {}
        for param in PARAMS:
            if param in ["n_estimators", "max_depth"]:
                new_params[param] = int(float(params[param]))
            else:
                new_params[param] = float(params[param])

        model = LGBMRegressor(**new_params)
        model.fit(X_train_vec, np.log1p(y_train))
        y_pred = np.exp(model.predict(X_val_vec))

        mlflow.log_metrics({
            "val_rmse": root_mean_squared_error(y_val, y_pred),
            "val_r2": r2_score(y_val, y_pred),
            "val_msle": mean_squared_log_error(y_val, y_pred)
        })

        mlflow.log_params(new_params)
        signature = infer_signature(X_val_vec, y_pred)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_val_vec[:2]
        )

@task(name="train_with_experiment_tracking")
def run_register_model(X_train_vec, y_train, X_val_vec, y_val, top_n=5):
    optimization_experiment = client.get_experiment_by_name(OPTIMIZATION_EXPERIMENT)
    runs = client.search_runs(
        experiment_ids=optimization_experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )

    for run in runs:
        train_and_log_model(
            X_train_vec, y_train, X_val_vec, y_val, run.data.params
        )

    #mlflow.log_artifact(temp_cloudpickle_filename, artifact_path="DictVectorizer")
    #os.remove(temp_cloudpickle_filename)

    current_experiment = client.get_experiment_by_name(CURRENT_EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=current_experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=["metrics.val_rmse ASC"]
    )[0]

    model_uri = f"runs:/{best_run.info.run_id}/model"
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=CURRENT_EXPERIMENT_NAME
    )
    logger.info(f"✅ Model successfully registered")

    client.transition_model_version_stage(
        name=CURRENT_EXPERIMENT_NAME,
        version=registered_model.version,
        stage="Production",
        archive_existing_versions=True
    )
    logger.info(f"✅ Model version {registered_model.version} transitioned to Production")

@task(name="train_register_best_model")
def train_register_best_model():
    df = read_pickle_from_s3(BUCKET, f"{PREFIX}/{pickle_file_name}")
    X_train_vec, y_train, X_val_vec, y_val, dv = preprocess_for_train(df)
    run_register_model(X_train_vec, y_train, X_val_vec, y_val)

if __name__ == '__main__':
    train_register_best_model()
