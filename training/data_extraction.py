import os
import shutil
import zipfile
import pandas as pd
from pathlib import Path

def download_and_extract_kaggle_data():
    """
    Downloads and extracts the Bike Sharing Demand dataset using Kaggle CLI.
    Assumes `kaggle.json` is placed in ~/.kaggle.
    """
    print("🚀 Downloading dataset from Kaggle...")
    os.system("kaggle competitions download -c bike-sharing-demand")

    zip_path = "bike-sharing-demand.zip"
    extract_path = Path("data/input_bike_data")
    extract_path.mkdir(parents=True, exist_ok=True)

    print("📦 Extracting ZIP...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    os.remove(zip_path)
    print("✅ Data extracted to:", extract_path)

    return extract_path


def load_data(input_dir: Path):
    """
    Loads train and test CSVs as DataFrames.
    """
    train_path = input_dir / "train.csv"
    test_path = input_dir / "test.csv"

    print("📄 Loading train and test data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    return train, test


def cleanup_temp_data(input_dir: Path):
    """
    Removes the temporary input_bike_data directory.
    """
    print("🧹 Cleaning up temporary files...")
    shutil.rmtree(input_dir, ignore_errors=True)
    print("✅ Temporary files deleted.")


def extract_data():
    """
    Main function to orchestrate data download, load, and cleanup.
    Returns:
        train (pd.DataFrame), test (pd.DataFrame)
    """
    input_dir = download_and_extract_kaggle_data()
    train, test = load_data(input_dir)
    cleanup_temp_data(input_dir)

    return train, test


# Optional: allow standalone execution
if __name__ == "__main__":
    train_df, test_df = extract_data()
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
