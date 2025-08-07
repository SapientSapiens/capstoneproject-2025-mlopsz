import os
import pickle
import cloudpickle
import logging
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts
from prefect import task

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

EXPERIMENT_NAME_PREFIX = "train_register_promote_best_model_around"
mlflow.set_tracking_uri("http://127.0.0.1:5000")


@task(name="retrieve_from_model_artifact_store_at_S3", retries=3, retry_delay_seconds=10)
def download_deployable_model(experiment_name, best_run_id):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise Exception(f"Experiment '{experiment_name}' not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.val_rmse ASC"],
        max_results=1
    )

    if not runs:
        raise Exception("No runs found in the experiment.")

    best_run = runs[0]
    run_id = best_run.info.run_id

    # verifying if they are same
    if best_run_id != run_id:
      logger.warning(f"Mismatch in model URIs — expected: {best_run_id}, found: {run_id}")
      return

    model_uri = f"runs:/{run_id}/model"
    logger.info(f"Model URI: {model_uri} matches")
    print(f"✅✅✅ Model URI: {model_uri} matches ✅✅✅")
    try:
        local_path = download_artifacts(
            run_id=run_id,
            artifact_path="model/model.pkl",
            dst_path="saved_model"
        )
        logger.info(f"Model artifacts downloaded to: {local_path}")
    except Exception as e:
        logger.error(f"Error downloading model artifacts: {e}")


@task(name="bundle_model_arifact_and_cleanup_files")
def bundle_and_cleanup():
    # Get the project root 
    project_root = Path(__file__).resolve().parent.parent

    dv_path = project_root / "training" / "dv.pkl"
    model_path = project_root / "saved_model" / "model" / "model.pkl"
    bundle_path = project_root / "deployment" / "deployed_models" / "model_bundle.pkl"

     # Load encoder and model
    with open(dv_path, "rb") as f_dv:
        dv = pickle.load(f_dv)

    with open(model_path, "rb") as f_model:
        model = pickle.load(f_model)

    model_bundle = {
        "model": model,
        "dv": dv
    }

    # Save bundled model to absolute path
    with open(bundle_path, "wb") as f_out:
        cloudpickle.dump(model_bundle, f_out)

    logger.info(f"✅ Model bundle saved to: {bundle_path}")
    print(f"✅✅✅ Model bundle saved to: {bundle_path} ✅✅✅")

    # Cleanup original files
    dv_path.unlink(missing_ok=True)
    model_path.unlink(missing_ok=True)
     
    # Remove saved_model directory if it's empty
    try:
        model_dir = model_path.parent
        if not any(model_dir.iterdir()):
            model_dir.rmdir()  # Remove "saved_model/model"
            model_dir.parent.rmdir()  # Remove "saved_model"
            logger.info("🧹 Removed empty saved_model directory.")
    except Exception as e:
        logger.warning(f"⚠️ Could not remove saved_model directory: {e}")

    logger.info("🧹 Cleaned up dv.pkl and model.pkl")

@task(name="prepare_production_artifacts")
def prepare_production_artifacts(best_rmse_on_hypertuning: str, best_run_id: str, version_number: str):
    if not best_rmse_on_hypertuning:
       raise ValueError("Best RMSE value is required but not provided.")
    if not best_run_id:
       raise ValueError("Best Run id value is required but not provided.")
    if not version_number:
       raise ValueError("Version number is required but not provided.")

    try:
        logger.info("⬇️  Downloading deployable model...")
        current_experiment_name =  f"Iteration-{version_number}_{EXPERIMENT_NAME_PREFIX}_{best_rmse_on_hypertuning}"
        download_deployable_model(current_experiment_name, best_run_id)
        logger.info("✅ Model downloaded successfully.\n")
    except Exception as e:
        logger.error(f"❌ Failed to download model: {e}")
        return

    try:
        logger.info("📦 Bundling model and encoder...")
        bundle_and_cleanup()
        logger.info("✅ Bundle created and cleanup done.\n")
    except Exception as e:
        logger.error(f"❌ Failed to bundle model and cleanup: {e}")


#if __name__ == '__main__':
   # prepare_production_artifacts()