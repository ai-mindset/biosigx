""" """

# %% [markdown]
# ## Imports

# %%
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

# %% [markdown]
# ## Plot functions


# %%
def plot_anomalies(
    original_signal: npt.NDArray[np.float64],
    anomalies: npt.NDArray[np.float64],
    sequence_length: int,
    title: str = "Anomaly Detection Results",
    figsize: tuple[int, int] = (15, 5),
) -> None:
    """Plot original signal with highlighted anomalies.

    Args:
        original_signal: Original time series data
        anomalies: Boolean array of detected anomalies
        sequence_length: Length of sequences used for detection
        title: Plot title
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)

    # Plot original signal
    plt.plot(original_signal, label="Signal", color="blue", alpha=0.7)

    # Highlight anomalous regions
    anomaly_indexes = np.where(anomalies)[0]
    for idx in anomaly_indexes:
        plt.axvspan(idx, idx + sequence_length, color="red", alpha=0.2)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# %%
def plot_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray,
    sequence_idx: int = 0,
    figsize: tuple[int, int] = (15, 5),
) -> None:
    """Plot original vs reconstructed sequence.

    Args:
        original: Original sequence
        reconstructed: Reconstructed sequence
        sequence_idx: Index of sequence to plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    plt.plot(original[sequence_idx], label="Original", color="blue", alpha=0.7)
    plt.plot(reconstructed[sequence_idx], label="Reconstructed", color="red", alpha=0.7)

    plt.title("Original vs Reconstructed Sequence")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
