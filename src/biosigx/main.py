"""Load, preprocess, train, and detect anomalies in ECG data using an autoencoder."""

# %% [markdown]
# ## Imports

# %%
import time

import numpy.typing as npt
from pandas import DataFrame, read_csv

from biosigx.data import load_and_preprocess_data
from biosigx.logger import setup_logger
from biosigx.plot import plot_anomalies
from biosigx.train import train_anomaly_detector
from biosigx.utils import detect_anomalies

# %% [markdown]
# ## Logging

# %%
log = setup_logger("INFO")


# %% [markdown]
# ## Main


# %%
def main(data: DataFrame, seq_length: int = 100, n_features: int = 139) -> npt.NDArray:
    """Load ECG data from a CSV file, preprocess it, train an anomaly detector, and detect anomalies in the test set.

    Args:
        data (DataFrame): DataFrame containing time-series data
        seq_length (int): Length of each sequence for the time series data.
        Default is 100.
        n_features (int): Number of features in each sequence. Default is 1.

    Returns:
        npt.NDArray: A numpy array containing the detected anomalies in the test set.

    """
    if data.empty:
        raise FileNotFoundError("Data were not loaded!")

    # Preprocess
    log.info("Preprocessing data...\n")
    X_train, X_test, _, _ = load_and_preprocess_data(data=data, seq_length=seq_length)
    log.info("Data were preprocessed into train and test arrays\n")

    # Train model
    log.info("Training model...\n")
    autoencoder, _, _ = train_anomaly_detector(
        X_train=X_train, seq_length=seq_length, n_features=n_features
    )
    log.info("Autoencoder was successfully trained\n")

    # Detect anomalies
    log.info("Detecting anomalies...\n")
    anomalies = detect_anomalies(autoencoder, X_test)
    log.info(f"{anomalies}\n")

    return anomalies


# %% [markdown]
# ## Main guard

# %%
if __name__ == "__main__":
    start_time = time.time()

    data_file: str = "data/ECG_data.csv"
    # Load your data
    df = read_csv(filepath_or_buffer=data_file, index_col=0, header=None)
    anomalies = main(data=df)

    end_time = time.time()
    elapsed_time = end_time - start_time

    log.info(f"Execution time of 'main()' function: {elapsed_time:.4f} seconds")

    # Plot results
    plot_anomalies(
        original_signal=df.values,
        anomalies=anomalies,
        sequence_length=100,
        title="Time Series Anomaly Detection",
    )
