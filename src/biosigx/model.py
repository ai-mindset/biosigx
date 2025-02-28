"""
Model architecture module for time series anomaly detection.

This module provides functions for building and evaluating autoencoder models
that detect anomalies in time series data by learning normal patterns and
identifying deviations.
"""

# %% [markdown]
# ## Autoencoder Model Architecture
#
# This section defines the convolutional autoencoder architecture for time series
# anomaly detection. The model learns to compress and reconstruct normal patterns
# in the data, which allows it to identify anomalies as sequences it fails to
# reconstruct accurately.

# %%
import keras
import numpy as np
from keras import layers
from numpy.typing import NDArray


# %%
def build_model(
    sequence_length: int,
    n_features: int = 1,
) -> keras.Model:
    """Build 1D convolutional autoencoder for time series anomaly detection.

    Args:
        sequence_length: Length of input sequences
        n_features: Number of features in each time step

    Returns:
        Compiled Keras model

    >>> model = build_model(sequence_length=288)
    >>> isinstance(model, keras.Model)
    True
    >>> model.input_shape
    (None, 288, 1)

    """
    model = keras.Sequential(
        [
            layers.Input(shape=(sequence_length, n_features)),
            layers.Conv1D(32, 7, padding="same", strides=2, activation="relu"),
            layers.Dropout(0.2),
            layers.Conv1D(16, 7, padding="same", strides=2, activation="relu"),
            layers.Conv1DTranspose(16, 7, padding="same", strides=2, activation="relu"),
            layers.Dropout(0.2),
            layers.Conv1DTranspose(32, 7, padding="same", strides=2, activation="relu"),
            layers.Conv1DTranspose(n_features, 7, padding="same"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
    )
    return model


# %%
def compute_threshold(
    model: keras.Model,
    train_data: NDArray[np.float64],
    percentile: float = 95,
) -> float:
    """Compute anomaly threshold from reconstruction error distribution.

    Args:
        model: Trained autoencoder model
        train_data: Training data used to establish normal behavior
        percentile: Percentile to use for threshold computation

    Returns:
        Threshold value for anomaly detection

    >>> import numpy as np
    >>> import keras
    >>> model = keras.Sequential([keras.layers.Lambda(lambda x: x)])
    >>> model.compile(loss="mse")
    >>> data = np.zeros((10, 5, 1))
    >>> data[0, 0, 0] = 1.0  # Create one outlier
    >>> threshold = compute_threshold(model, data, percentile=90)
    >>> threshold > 0
    True

    """
    predictions = model.predict(train_data)
    errors = np.mean(np.abs(predictions - train_data), axis=(1, 2))
    return np.percentile(errors, percentile)


# %% [markdown]
# ## Module Testing
#
# This section allows testing the model module independently.

# %%
if __name__ == "__main__":
    # Test model building
    sequence_length = 288
    model = build_model(sequence_length)
    model.summary()

    # Test threshold computation with dummy data
    dummy_data = np.random.normal(size=(100, sequence_length, 1))
    threshold = compute_threshold(model, dummy_data)
    print(f"Computed threshold: {threshold:.6f}")
