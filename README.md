# 🚲Bike Sharing Demand Hourly Prediction

 ## 🧩Problem Description

  Bike sharing systems offer a flexible and eco-friendly way to navigate urban environments 🌆🚲. Users can rent a bike from one station and return it to another, thanks to automated kiosks spread across cities. With over 500+ programs worldwide, these systems have become vital to sustainable urban mobility 🌍♻️.

  But running them efficiently is a challenge.

  🔢 Operators must anticipate hourly fluctuations in bike demand


  ➡️ To ensure bikes are available when and where they’re needed


  ➡️ To prevent overcrowded or empty docking stations


  ➡️ To avoid resource wastage and customer dissatisfaction


  📉 Inaccurate demand forecasts lead to poor user experience and operational inefficiencies


  📊 The demand varies due to multiple factors like time of day, weather, season, or public events


  Now, the technical twist:


  ⚠️ Even well-performing ML models tend to degrade over time


  📉 This is caused by data drift — shifts in user behavior, weather trends, and unexpected anomalies (e.g., local festivals or global events like a pandemic)


  ❌ Without proper monitoring and updating, these models can make unreliable predictions that hurt business outcomes


  That’s why the challenge isn’t just about forecasting demand — it’s about keeping predictions accurate consistently, despite change 🌀



 ## 🧠Solution Outline

  This project presents a machine learning-based solution to forecast hourly bike rental demand. The approach involves combining historical usage data with external features like weather conditions and time-based variables to train a predictive model capable of estimating future demand.

  Beyond model accuracy, the project also implements MLOps practices to ensure the solution is scalable, maintainable, and reliable over time. This includes:

  🔄 Automated data preprocessing and feature engineering

  🧪 Experiment tracking for model comparison and reproducibility

  📦 Model versioning and registry for production readiness

  🌐 Deployment of the model as a robust, live prediction service

  📊 Scheduled model retraining workflows and continuous monitoring of drifts

  The result is not just a high-performing predictive model, but a __sustainable machine learning operation (MLOPS)__ pipeline that supports continuous improvements and long-term business value.



 ## 📦Data Overview

  This project uses real-world data from the Capital Bikeshare system in Washington, D.C., covering hourly rental activity from 2011 to 2012 🗓️🚲. To build a unified and comprehensive dataset, data was ingested from two different sources:


   1️⃣ Kaggle - Bike Sharing Demand

   🔹 Contains actual (unscaled) feature values

   🔹 Test set here does not include the target variable (count)

   2️⃣ UCI - Bike Sharing Dataset

   🔹 Contains normalized feature values

   🔹 Provides the target variable even for the test data

   To overcome the limitations of both datasets, we merged them thoughtfully to produce a single, rich dataset suitable for training and evaluation. This data transformation and unification simulates a simplified ETL pipeline, where:

   📥 Data was Extracted from two sources

   🧹 Cleaned, Transformed to standard structure

   🪣 Loaded into a simulated Data Lake (S3) for downstream usage ------->>> the training pipiline!!


   ### 🧾Dataset Summary

   The dataset includes hourly data points with associated weather, time, and user metrics:

   - *datetime – timestamp of the rental*

   - *season – 1: spring 🌱 | 2: summer ☀️ | 3: fall 🍂 | 4: winter ❄️*

   - *holiday – whether the day was a public holiday 🎉*

   - *workingday – true if it was a weekday that's not a holiday 📆*

   - *weather – encoded weather condition (clear, mist, snow, etc.) 🌦️*

   - *temp, atemp – actual and “feels like” temperature 🌡️*

   - *humidity, windspeed – relative humidity and wind speed 💨*

   - *casual, registered – rental counts for unregistered and registered users*

   - *count – total rental count per hour (📌 our target variable)*


   ### 📚Data Splitting Strategy

   To simulate a realistic MLOps lifecycle, the 24-month dataset is split as follows:

   _📊 Train Set (Months 1–16)_

   → Used to train the initial model (primarily data from 2011)

   _🧪 Validation/Test Set (Months 17–20)_

   → Used to evaluate model generalization and performance

   → Also serves as the reference dataset in monitoring workflows (e.g., with Evidently AI)

   _📈 Simulated Production Data (Months 21–24)_

   → Acts as "newly arriving" data with ground truth labels

   → Used to detect drift, trigger retraining if needed, and simulate batch ingestion


   #### This structured setup mirrors a production-ready data pipeline and ensures the project aligns with MLOps principles — preparing us not just to build a good model, but to monitor and maintain it continuously ✅




 ## ⚙️🔧Technical Overview

   ### Significant Technology Stack I have used ###

   - Amazon Web Service as the main cloud platform

   - S3 bucket as the data lake

   - Terraform as Infrastructure as Code (IaC) tool.

   - Prefect as the orchestration tool scheduling and executing pipeline workflows

   - MLFlow for experiment tracking, moodel versioning and artifact storing at S3

   - Evidently AI for MLOPS monitoring & drift detection, and  Graphana for isualiztion & alerting (__**to be completed**__)

   - Hyperopt, scikit-learn, LightGBM  for ML tasks

   - Uvicorn and FastAPI for model serving

   - Docker & docker-compose for containerization & container orchestration

   - Streamlit to build interactive UI for frontent interface for prediction  service



 ## ☁️Cloud and Terraform

  - This project is entirely developed on AWS Cloud. Only the infrastructure provisioning at AWS has been done from my local machine.

  - From my local machine, I used Terraform as the IaC toolto create


  ## 📦 ⟶ 🔁 ⟶ 🎯Reporoducibility ##

   #### 🏭Kindly set up the environment and configuration of the VM or your local machine (with WSL). Sequentially proceeed : ####

   - Installing annaconda

            wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh

            bash Anaconda3-2024.10-1-Linux-x86_64.sh

   - Install docker

            update apt before doing so

            sudo apt update   

            sudo apt install docker.io

            sudo gpasswd -a $USER docker

            sudo service docker restart

     loguot and re-login into the VM to this take effect

  - Install docker-compose

    create a directory bin in the home directory of the VM and get inside the same

            mkdir bin

            cd bin

    download docker-compose and make it executable

            wget https://github.com/docker/compose/releases/download/v2.34.0/docker-compose-linux-x86_64 -O docker-compose

            chmod +x docker-compose

    return to home directory and add the path to the bin directory to the PATH variable in .bashrc

             cd ~

            nano .bashrc

            export PATH="${HOME}/bin:${PATH}"  # add this line at the end of the .bashrc file. Save and exit the nano editor.

            source .bashrc


   - Git clone this repository 

            git clone https://github.com/SapientSapiens/capstoneproject-2025-mlopsz.git


   - Create conda virtual environment for this project (i shall use my env name here and I used python version 3.11.5 in the project)

            conda create -n capstoneproject-2025-mlopsz-env python=3.11.5

   - Activate conda virtual environment

            conda activate capstoneproject-2025-mlopsz-env

   - Move indide the cloned project repository and install the project dependencies

            cd  capstoneproject-2025-mlopsz-env

            pip install -r requirements.txt


   - Set up S3 bucket on aws manually with awscli or through the AWS webc console, or automate with Terraform. For aws, you need to your EC2 instance has permission to use the bucket. For your local machine, you can use localstack

            aws s3 mb s3://<your-bucket-name>


   #### 🏗️ Now we can run the trainig pipeline : ####

   - Start the MLFlow Server. We assume you are in the project root for all commands. Also note my artifact store is pointing to my S3 bucket, you need to set it accordingly.

            ./training/start_mlflow.sh

   - Start the Prefect server. 

            prefect server start

   - You can also pull up the Prefect and MLFlow UI at http://localhost:4200 and http://localhost:5000. If you are running it on a VM, ensure port forward for these port are enabled.

   - Now you can run the orchestrated training pipeline for the first time. 

            python -m orchestration.orchestrated_training_pipeline --run-now

   - Then you can deploy the training pipeline orchestrated workflow to Prefect. It shall execute automatically at the scheduled (set with a cron expression in the main flow) time.

             python -m orchestration.orchestrated_training_pipeline

   
   #### 🚀Activate the deployment service pipeline : ####

   - One of the output of running the training pipeline is you that get the model artifact bundle (model + encoder) deployed to the deploymwnt service pipeline (/deployment/deployed_models/).

   - So now we only have to spin up the containers (one for the model serving and the other for the Streamlit model inference app) in a orchestrated way with docker-compose. 

             ./deploymwnt/activate_deployment.sh

  - Once the services/containers are up, you can invoke http://<your-public-ip-or-your-localhost>:8501 and use the Bike Sharing Demand Hourly Prediction Service.


      


   