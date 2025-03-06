"""Data processing module for time series anomaly detection.

This module provides functions for loading, preprocessing, and preparing
time series data for anomaly detection. It handles data normalization and
sequence creation for training autoencoder models.
"""

# %% [markdown]
# ## Data Loading and Processing
#
# This section contains functions for loading raw time series data and
# converting it into properly formatted sequences for model training.

# %%
import ipdb
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler

from biosigx.logger import setup_logger

# %% [markdown]

# ## Logger Setup
# %%
logger = setup_logger()


# %%
def load_data(data_file: str) -> pd.DataFrame:
    """Load timeseries data from a file.

    Args:
        data_file (str): Path to the CSV file containing time series data

    Returns:
        pd.DataFrame: DataFrame with loaded time series data

    Example:
        >>> import tempfile
        >>> # Create a temporary file and write some CSV data to it
        >>> temp = tempfile.NamedTemporaryFile(delete=False, mode="w+t")
        >>> _ = temp.write("1.0,2.0,3.0\\n4.0,5.0,6.0")
        >>> temp.flush()
        >>> df = load_data(temp.name)
        >>> df.shape[0]
        2
        >>> # Clean up the temporary file after use
        >>> os.unlink(temp.name)

    """
    return pd.read_csv(data_file, header=None)


# %%
def prepare_sequences(
    data: NDArray[np.float64],
    sequence_length: int = 288,
    stride: int = 1,
) -> NDArray[np.float64]:
    """Create sequences from time series data for training/inference.

    Args:
        data: Time series data array
        sequence_length: Length of each sequence
        stride: Step size between consecutive sequences

    Returns:
        Array of sequences shaped (n_sequences, sequence_length, n_features)

    >>> import numpy as np
    >>> data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    >>> sequences = prepare_sequences(data, sequence_length=3, stride=1)
    >>> sequences.shape
    (3, 3, 1)

    """
    sequences = []
    for i in range(0, len(data) - sequence_length + 1, stride):
        sequence = data[i : i + sequence_length]
        sequences.append(sequence)
    return np.array(sequences)


# %%
def process_data(
    data: pd.DataFrame,
    sequence_length: int = 288,
    scaler: StandardScaler | None = None,
) -> tuple[NDArray[np.float64], StandardScaler]:
    """Process raw data into model-ready sequences.

    Args:
        data: Raw time series data
        sequence_length: Length of each sequence
        scaler: Optional pre-fitted StandardScaler for normalization

    Returns:
        Tuple of (normalized sequences, fitted scaler)

    >>> import numpy as np
    >>> import pandas as pd
    >>> data = pd.DataFrame(np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]))
    >>> sequences, scaler = process_data(data, sequence_length=3)
    >>> sequences.shape
    (3, 3, 1)
    >>> abs(scaler.mean_[0] - 3.0) < 0.01
    True

    """
    if scaler is None:
        scaler = StandardScaler()
        normalized = scaler.fit_transform(data)
    else:
        normalized = scaler.transform(data)

    sequences = prepare_sequences(normalized, sequence_length)
    return sequences, scaler


# %% [markdown]
# ## Module Testing
#
# This section allows testing the data module independently.

# %%

if __name__ == "__main__":
    # Testing with sample data in a tempfile
    sample_data = "1.0,2.0,3.0\n4.0,5.0,6.0\n7.0,8.0,9.0"

    import tempfile

    # Create a temporary file and write the sample data to it
    with tempfile.NamedTemporaryFile(mode="w+t", delete=False) as temp_file:
        temp_file.write(sample_data)
        temp_file_path = temp_file.name

    # Load data from the temporary file
    df = load_data(temp_file_path)
    logger.info(f"Loaded data shape: {df.shape}")

    sequences, scaler = process_data(df, sequence_length=2)
    logger.info(f"Processed sequences shape: {sequences.shape}")
    logger.info(f"Scaler mean: {scaler.mean_}")
    logger.info(f"Scaler scale: {scaler.scale_}")

    # Optionally, you can remove the temporary file after use
    import os

    os.remove(temp_file_path)
