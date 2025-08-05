from prefect import flow

# Import the @task-decorated functions directly from the scripts
from training.feature_engineering import feature_engineering_main
from training.train_experiment import run_track_experiment
from training.register_best_model import train_register_best_model
from training.deploy_production_artifacts import prepare_production_artifacts


@flow(name="orchestrated-bike-sharing-train-pipeline")
def orchestrated_training_pipeline():
    feature_engineering_main()         # Step 1: Feature engineering
    run_track_experiment()            # Step 2: MLflow training experiment
    train_register_best_model()       # Step 3: Register the best model
    prepare_production_artifacts()    # Step 4: Bundle for production


if __name__ == "__main__":
    import sys

    if "--run-now" in sys.argv:
        orchestrated_training_pipeline()  # Run immediately
    else:
        orchestrated_training_pipeline.serve(
            name="bike-sharing-train-pipeline",
            cron="0 2 1 */3 *",
            tags=["bike-sharing", "training-pipeline"],
        )
