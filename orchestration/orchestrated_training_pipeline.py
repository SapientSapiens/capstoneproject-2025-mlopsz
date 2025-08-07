from prefect import flow

# Import the @task-decorated functions directly from the scripts
from training.feature_engineering import feature_engineering_main
from training.train_experiment import run_track_experiment
from training.register_best_model import train_register_best_model
from training.deploy_production_artifacts import prepare_production_artifacts

VERSION_NUMBER = "2.2"

@flow(name="orchestrated-bike-sharing-demand-prediction-train-pipeline")
def orchestrated_training_pipeline():
    feature_engineering_main()                                                                              # Step 1: Feature engineering
    best_rmse_on_hypertuning = run_track_experiment(VERSION_NUMBER)                                         # Step 2: MLflow optimized hype-paramenter search experiment
    best_run_id = train_register_best_model(best_rmse_on_hypertuning, VERSION_NUMBER)                       # Step 3: Train, Register & Promote the best model
    prepare_production_artifacts(best_rmse_on_hypertuning, best_run_id, VERSION_NUMBER)                     # Step 4: Bundle for production
    

if __name__ == "__main__":
    import sys

    if "--run-now" in sys.argv:
        orchestrated_training_pipeline()  # Run immediately
    else:
        orchestrated_training_pipeline.serve(
            name="bike-sharing-train-pipeline",
            cron="0 0 1 */3 *",
            tags=["bike-sharing-demand-prediction", "training-pipeline"],
        )