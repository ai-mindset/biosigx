"""Module for detecting anomalies in data using a trained autoencoder model."""

# %%
import numpy as np
import numpy.typing as npt
from keras import Model


# %%
def detect_anomalies(
    model: Model, data: npt.NDArray[np.float64], threshold_sigma: float = 3.0
) -> npt.NDArray[np.float64]:
    """Detect anomalies using reconstruction error.

    Args:
        model: Trained autoencoder model
        data: Input data to check for anomalies
        threshold_sigma: Number of std deviations for anomaly threshold

    Returns:
        Binary array indicating anomalies

    """
    # Get reconstruction error
    reconstructed = model.predict(data)
    mse = np.mean(np.power(data - reconstructed, 2), axis=(1, 2))

    # Find threshold
    threshold = np.mean(mse) + threshold_sigma * np.std(mse)

    # Detect anomalies
    anomalies = mse > threshold

    return anomalies
