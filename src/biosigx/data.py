"""Scale 3D sequence data and load/prepare time series data for anomaly detection."""


# %% [markdown]
# ## Imports

# %%
from typing import cast

import numpy as np
import numpy.typing as npt
from pandas import DataFrame, read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %%


# %% [markdown]
# ## Scale


# %%
def scale_sequences(
    X_train: npt.NDArray[np.float64], X_test: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Scale 3D sequence data using StandardScaler.

    Args:
        X_train: Training sequences with shape (n_sequences, seq_length, n_features)
        X_test: Test sequences with shape (n_sequences, seq_length, n_features)

    Returns:
        Scaled training and test sequences with original shapes

    """
    # Store original shapes
    train_shape = X_train.shape
    test_shape = X_test.shape

    # Reshape to 2D: (n_sequences * seq_length, n_features)
    X_train_2d = X_train.reshape(-1, train_shape[-1])
    X_test_2d = X_test.reshape(-1, test_shape[-1])

    # Fit scaler on training data and transform both sets
    scaler = StandardScaler()
    X_train_scaled_2d = cast(npt.NDArray[np.float64], scaler.fit_transform(X_train_2d))
    X_test_scaled_2d = cast(npt.NDArray[np.float64], scaler.transform(X_test_2d))

    # Reshape back to 3D
    X_train_scaled_3d = cast(
        npt.NDArray[np.float64], X_train_scaled_2d.reshape(train_shape)
    )
    X_test_scaled_3d = cast(npt.NDArray[np.float64], X_test_scaled_2d.reshape(test_shape))

    return X_train_scaled_3d, X_test_scaled_3d


# %% [markdown]
# ## Load and preprocess data


# %%
def load_and_preprocess_data(
    data: DataFrame,
    seq_length: int = 140,
    test_size: float = 0.2,
    random_state: int | None = 42,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Load and preprocess time series data for anomaly detection.

    Args:
        data: Raw time series data
        seq_length: Length of sequences to create
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test arrays

    Raises:
        ValueError

    """
    if data.empty:
        raise ValueError("You are attempting to preprocess an empty dataframe")

    # Create sequences
    sequences = [
        window.values for window in data.rolling(window=seq_length, min_periods=1)
    ]
    # Remove any sequences that are shorter than seq_length
    sequences = [seq for seq in sequences if len(seq) == seq_length]

    X = np.array(sequences)
    y = np.zeros(len(X))  # Binary labels: 0 for normal, 1 for anomaly

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale the data
    X_train = cast(npt.NDArray[np.float64], X_train)
    X_test = cast(npt.NDArray[np.float64], X_test)
    y_train = cast(npt.NDArray[np.float64], y_train)
    y_test = cast(npt.NDArray[np.float64], y_test)
    X_train, X_test = scale_sequences(X_train=X_train, X_test=X_test)

    return X_train, X_test, y_train, y_test


# %% [markdown]
# ## Main guard

# %%
if __name__ == "__main__":
    data = read_csv("data/ECG_data.csv", index_col=0, header=None)
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data=data)
