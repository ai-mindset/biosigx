"""Visualization module for time series anomaly detection.

This module provides functions for visualizing time series data, model
performance, and detected anomalies. It helps understand model behaviour
and interpret results.
"""

# %% [markdown]
# ## Data and Result Visualisation
#
# This section contains functions for visualizing time series data, model
# training history, and anomaly detection results.

# %%
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray


# %%
def plot_time_series(
    data: pd.DataFrame | NDArray[np.float64],
    anomalies: NDArray[np.bool_] | None = None,
    sequence_length: int | None = None,
    title: str = "Time Series Data",
    figsize: tuple[int, int] = (12, 6),
) -> None:
    """Plot time series data with optional anomaly highlighting.

    Args:
        data: Time series data to plot
        anomalies: Boolean array indicating which points/sequences are anomalies
        sequence_length: Length of sequences if anomalies are at sequence level
        title: Plot title
        figsize: Figure size (width, height)

    """
    plt.figure(figsize=figsize)

    # Convert data to pandas Series if it's a numpy array
    if isinstance(data, np.ndarray):
        data_series = pd.Series(data.flatten())
    else:
        data_series = data.iloc[:, 0]  # Assume first column contains values

    # Plot data
    plt.plot(data_series, label="Data", color="blue")

    # Highlight anomalies if provided
    if anomalies is not None:
        # Check if we need to convert sequence anomalies to point anomalies
        if sequence_length is not None and len(anomalies) != len(data_series):
            # Convert sequence anomalies to point anomalies
            point_anomalies = np.zeros(len(data_series), dtype=bool)
            for i, is_anomaly in enumerate(anomalies):
                if is_anomaly:
                    point_anomalies[i : i + sequence_length] = True
            anomaly_indices = np.where(point_anomalies)[0]
        elif len(anomalies) == len(data_series):
            # Anomalies already at point level
            anomaly_indices = np.where(anomalies)[0]
        else:
            raise ValueError(
                f"Length mismatch: {len(anomalies)} anomalies vs {len(data_series)} data points. "
                f"If these are sequence anomalies, provide sequence_length."
            )

        plt.scatter(
            anomaly_indices,
            data_series.iloc[anomaly_indices]
            if isinstance(data, pd.DataFrame)
            else data_series[anomaly_indices],
            color="red",
            label="Anomalies",
        )

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()


# %%
def plot_training_history(
    history: keras.callbacks.History,
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """Plot model training loss history.

    Args:
        history: Keras model training history
        figsize: Figure size (width, height)

    Returns:
        None (shows plot)

    >>> import keras
    >>> model = keras.Sequential()
    >>> model.compile(optimizer="adam", loss="mse")
    >>> import numpy as np
    >>> data = np.random.random((10, 5))
    >>> history = model.fit(data, data, epochs=3, verbose=0)
    >>> plot_training_history(history)  # Shows plot

    """
    plt.figure(figsize=figsize)
    plt.plot(history.history["loss"], label="Training Loss")

    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Validation Loss")

    plt.title("Model Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# %%
def plot_reconstruction_error(
    errors: NDArray[np.float64],
    threshold: float,
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """Plot reconstruction error histogram with threshold.

    Args:
        errors: Array of reconstruction errors
        threshold: Anomaly detection threshold
        figsize: Figure size (width, height)

    Returns:
        None (shows plot)

    >>> import numpy as np
    >>> errors = np.random.normal(0, 1, 1000)
    >>> plot_reconstruction_error(errors, 2.0)  # Shows plot

    """
    plt.figure(figsize=figsize)
    plt.hist(errors, bins=50, alpha=0.8, density=True, label="Error Distribution")
    plt.axvline(
        threshold, color="red", linestyle="--", label=f"Threshold ({threshold:.4f})"
    )

    plt.title("Reconstruction Error Distribution")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# %%
def visualise_reconstruction(
    original: NDArray[np.float64],
    reconstructed: NDArray[np.float64],
    index: int = 0,
    figsize: tuple[int, int] = (12, 6),
) -> None:
    """Visualise original sequence and its reconstruction.

    Args:
        original: Original sequences
        reconstructed: Reconstructed sequences from the model
        index: Index of sequence to visualize
        figsize: Figure size (width, height)

    Returns:
        None (shows plot)

    >>> import numpy as np
    >>> original = np.sin(np.linspace(0, 10, 100)).reshape(1, 100, 1)
    >>> reconstructed = original * 0.9  # Slightly different reconstruction
    >>> visualise_reconstruction(original, reconstructed)  # Shows plot

    """
    plt.figure(figsize=figsize)
    plt.plot(original[index, :, 0], label="Original", color="blue")
    plt.plot(reconstructed[index, :, 0], label="Reconstructed", color="orange")

    plt.title(f"Sequence Reconstruction (Index: {index})")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ## Module Testing
#
# This section allows testing the visualization module independently.

# %%
if __name__ == "__main__":
    # Generate sample data
    n_points = 500
    time = np.linspace(0, 10, n_points)
    signal = np.sin(time) + 0.2 * np.random.randn(n_points)

    # Inject anomalies
    anomaly_indices = np.array([100, 200, 300, 400])
    signal[anomaly_indices] = signal[anomaly_indices] + 3

    # Create anomaly labels
    anomalies = np.zeros(n_points, dtype=bool)
    anomalies[anomaly_indices] = True

    # Visualise data with anomalies
    df = pd.DataFrame(signal)
    plot_time_series(df, anomalies, title="Sample Time Series with Anomalies")

    # Create mock reconstruction errors
    errors = np.abs(np.random.normal(0, 0.1, n_points))
    errors[anomaly_indices] = 0.6  # Make anomalies more visible

    # Visualise reconstruction errors
    plot_reconstruction_error(errors, 0.4)

    # Visualise a mock reconstruction
    original = np.sin(np.linspace(0, 6 * np.pi, 100)).reshape(1, 100, 1)
    reconstructed = original + 0.1 * np.random.randn(1, 100, 1)
    visualise_reconstruction(original, reconstructed)
