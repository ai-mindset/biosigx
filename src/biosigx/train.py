"""Model training module for time series anomaly detection.

This module provides functions for training autoencoder models on time series data
and using them to detect anomalies. It handles model training, threshold computation,
and anomaly detection.
"""

# %% [markdown]
# ## Model Training and Anomaly Detection
#
# This section contains functions for training the autoencoder model and using it
# to detect anomalies in time series data.

# %%
import keras
import numpy as np
from numpy.typing import NDArray

from biosigx.logger import setup_logger
from biosigx.model import compute_threshold

# %% [markdown]
# ## Logger Setup


# %%
logger = setup_logger()


# %%
def train_model(
    model: keras.Model,
    train_data: NDArray[np.float64],
    validation_split: float = 0.1,
    batch_size: int = 128,
    epochs: int = 50,
    patience: int = 5,
) -> tuple[keras.Model, float]:
    """Train model and compute anomaly threshold.

    Args:
        model: Autoencoder model to train
        train_data: Training data
        validation_split: Fraction of data to use for validation
        batch_size: Training batch size
        epochs: Maximum number of epochs
        patience: Early stopping patience

    Returns:
        Tuple of (trained model, anomaly threshold)

    >>> import numpy as np
    >>> import keras
    >>> model = keras.Sequential([keras.layers.Dense(1, input_shape=(5, 1))])
    >>> model.compile(optimizer="adam", loss="mse")
    >>> data = np.zeros((10, 5, 1))
    >>> trained_model, threshold = train_model(model, data, epochs=1)
    >>> isinstance(trained_model, keras.Model)
    True
    >>> threshold >= 0
    True

    """
    history = model.fit(
        x=train_data,
        y=train_data,  # In an autoencoder, we're teaching the model to copy the input perfectly
        validation_split=validation_split,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                mode="min",
                restore_best_weights=True,
            ),
        ],
        verbose=1,
    )

    threshold = compute_threshold(model, train_data)
    return model, threshold


# %%
def detect_anomalies(
    model: keras.Model,
    data: NDArray[np.float64],
    threshold: float,
) -> NDArray[np.bool_]:
    """Detect anomalies in data using trained model.

    Args:
        model: Trained autoencoder model
        data: Data to check for anomalies
        threshold: Anomaly detection threshold

    Returns:
        Boolean array indicating anomalies

    >>> import numpy as np
    >>> import keras
    >>> model = keras.Sequential([keras.layers.Lambda(lambda x: x)])
    >>> model.compile(loss="mse")
    >>> data = np.zeros((5, 3, 1))
    >>> data[0] = 1.0  # Make first sequence anomalous
    >>> anomalies = detect_anomalies(model, data, threshold=0.5)
    >>> anomalies[0]
    True
    >>> anomalies[1]
    False

    """
    predictions = model.predict(data, verbose=0)
    errors = np.mean(np.abs(predictions - data), axis=(1, 2))
    return errors > threshold


# %% [markdown]
# ## Module Testing
#
# This section allows testing the training module independently.

# %%
if __name__ == "__main__":
    import numpy as np
    from model import build_model
    from sklearn.preprocessing import StandardScaler

    # Create synthetic data
    n_samples = 1000
    sequence_length = 100
    synthetic_data = np.sin(np.linspace(0, 50, n_samples)).reshape(-1, 1)

    # Standardise data
    scaler = StandardScaler()
    normalised_data = scaler.fit_transform(synthetic_data)

    # Create sequences
    sequences = np.array(
        [
            normalised_data[i : i + sequence_length]
            for i in range(len(normalised_data) - sequence_length)
        ]
    )

    # Build and train model
    model = build_model(sequence_length)
    model, threshold = train_model(model, sequences, epochs=5)

    # Create test data with anomalies
    anomaly_sequences = sequences.copy()
    anomaly_sequences[0, :, 0] = 10  # Create an obvious anomaly

    # Detect anomalies
    anomalies = detect_anomalies(model, anomaly_sequences, threshold)
    logger.info(
        f"Detected {np.sum(anomalies)} anomalies out of {len(anomalies)} sequences"
    )
    logger.info(f"First sequence (should be anomalous): {anomalies[0]}")
