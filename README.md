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



 ## 🧠Solution Overview

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

   