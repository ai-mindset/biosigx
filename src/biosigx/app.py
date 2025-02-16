"""Biological signal processing and classification with TensorFlow.

Provides utilities for handling 1D biosignals (ECG, respiratory) including:
- Signal generation and validation
- Preprocessing and filtering
- CNN-based anomaly classification
- Interactive visualization

Requires: numpy, tensorflow, scipy, pydantic
"""

# %% Imports
from collections.abc import Callable
from dataclasses import dataclass

import keras
import matplotlib
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import numpy.typing as npt
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import BaseModel, Field, field_validator
from scipy import signal as sig
from sklearn.model_selection import load_model, train_test_split

plt.switch_backend("qtagg")
matplotlib.use("qtagg")

# https://github.com/neuropsychology/NeuroKit?tab=readme-ov-file#quick-example


# %% Data Models
class Signal1D(BaseModel):
    """Validated container for 1D biological signal data.

    Attributes:
        data: 1D array of signal samples
        sampling_rate: Sample frequency in Hz, must be positive
        signal_type: Signal category ('ECG', 'RESP', etc.)

    Raises:
        ValueError: If data contains NaN/Inf or sampling_rate ≤ 0

    """

    data: npt.NDArray[np.float64] = Field(..., description="Raw signal data")
    sampling_rate: float = Field(gt=0, description="Sampling rate in Hz")
    signal_type: str = Field(..., description="Type of biological signal")

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("data")
    def validate_data(cls, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Validate signal data array.

        Returns:
            NDArray[np.float64]: Validated signal data

        """
        if not np.isfinite(v).all():
            raise ValueError("Signal contains NaN or Inf values")
        return v


# %% Sample Data Generation
ecg = nk.ecg_simulate(duration=10, heart_rate=70)
# Process it
signals, info = nk.ecg_process(ecg, sampling_rate=250)

# Visualise the processing
nk.ecg_plot(signals, info)
rsp = nk.rsp_simulate(duration=60, sampling_rate=250, respiratory_rate=15)
# Process it
signals, info = nk.rsp_process(rsp, sampling_rate=250)

# Visualise the processing
nk.rsp_plot(signals, info)


def generate_ecg(duration_s: float, sampling_rate: float) -> Signal1D:
    """Generate synthetic ECG with possible arrhythmia.

    Args:
        duration_s: Signal duration in seconds
        sampling_rate: Sampling frequency in Hz

    Returns:
        Signal1D: Synthetic ECG signal

    Example:
        >>> ecg = generate_ecg(duration_s=1.0, sampling_rate=100)
        >>> len(ecg.data) == 100
        True

    """
    t = np.linspace(0, duration_s, int(duration_s * sampling_rate))
    normal = np.sin(2 * np.pi * 1.0 * t) * np.exp(-(((t % 1.0) - 0.1) ** 2) / 0.01)

    arrhythmia = np.random.normal(0, 0.1, len(t))
    signal = normal + arrhythmia

    return Signal1D(
        data=signal.astype(np.float64), sampling_rate=sampling_rate, signal_type="ECG"
    )


# %% Signal Processing
def preprocess_signal(
    signal: Signal1D, window_size: int = 1024
) -> npt.NDArray[np.float64]:
    """Preprocess 1D signal for model input.

    Args:
        signal: Input biological signal
        window_size: FFT window size

    Returns:
        NDArray[np.float64]: Preprocessed signal

    Example:
        >>> sig = Signal1D(data=np.ones(100), sampling_rate=100, signal_type="ECG")
        >>> processed = preprocess_signal(sig)
        >>> len(processed) == len(sig.data)
        True

    """
    # Normalize
    normalized = (signal.data - np.mean(signal.data)) / (np.std(signal.data) + 1e-8)

    # Define filter parameters
    nyquist = signal.sampling_rate / 2
    if signal.signal_type == "ECG":
        low, high = 0.5 / nyquist, 40 / nyquist
    elif signal.signal_type == "RESP":
        low, high = 0.1 / nyquist, 2 / nyquist
    else:
        raise ValueError(f"Unknown signal type: {signal.signal_type}")

    # Apply Butterworth bandpass
    b, a = sig.butter(4, [low, high], btype="band")
    filtered = sig.filtfilt(b, a, normalized, padlen=3 * (max(len(a), len(b)) - 1))

    return filtered


# %% Model Definition
def create_classifier(input_length: int, n_classes: int) -> keras.Model:
    """Create 1D CNN classifier for biological signals.

    Args:
        input_length: Length of input signal
        n_classes: Number of classification classes

    Returns:
        keras.Model: Compiled classifier model

    Example:
        >>> model = create_classifier(input_length=1000, n_classes=2)
        >>> model.input_shape == (None, 1000, 1)
        True

    """
    inputs = keras.Input(shape=(input_length, 1))

    x = keras.layers.Conv1D(32, kernel_size=16, activation="relu", padding="same")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2)(x)

    x = keras.layers.Conv1D(64, kernel_size=8, activation="relu", padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2)(x)

    x = keras.layers.Conv1D(64, kernel_size=4, activation="relu", padding="same")(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(n_classes, activation="softmax")(x)

    return keras.Model(inputs=inputs, outputs=outputs)


# %% Visualization
@dataclass
class PlotConfig:
    """Configuration for signal plotting."""

    title: str = ""
    xlabel: str = "Time (s)"
    ylabel: str = "Amplitude"
    figsize: tuple[int, int] = (10, 4)
    color: str = "blue"
    window_size: int = 500  # Points to show in animation window


def plot_signal(
    signal: Signal1D,
    config: PlotConfig | None = None,
) -> tuple[Figure, Axes]:
    """Plot a 1D signal.

    Args:
        signal: Signal to plot
        config: Plot configuration

    Returns:
        tuple[Figure, Axes]: matplotlib figure and axes

    Example:
        >>> sig = Signal1D(data=np.zeros(100), sampling_rate=100, signal_type="ECG")
        >>> fig, ax = plot_signal(sig)
        >>> isinstance(ax, Axes)
        True

    """
    config = config or PlotConfig()

    fig, ax = plt.subplots(figsize=config.figsize)
    t = np.arange(len(signal.data)) / signal.sampling_rate

    ax.plot(t, signal.data, color=config.color)
    ax.set_title(config.title or f"{signal.signal_type} Signal")
    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)
    ax.grid(True)

    return fig, ax


def animate_signal(
    signal: Signal1D,
    config: PlotConfig | None = None,
    update_func: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    | None = None,
) -> FuncAnimation:
    """Animate a 1D signal with optional real-time updates.

    Args:
        signal: Signal to animate
        config: Plot configuration
        update_func: Optional function to update signal values

    Returns:
        FuncAnimation: matplotlib animation object

    """
    config = config or PlotConfig()
    fig, ax = plt.subplots(figsize=config.figsize)
    (line,) = ax.plot([], [], color=config.color)

    # Set fixed axes limits
    t = np.arange(len(signal.data)) / signal.sampling_rate
    ax.set_xlim(0, config.window_size / signal.sampling_rate)
    y_min, y_max = float(np.min(signal.data)), float(np.max(signal.data))
    ax.set_ylim(y_min * 1.1, y_max * 1.1)

    ax.set_title(config.title or f"{signal.signal_type} Signal")
    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)
    ax.grid(True)

    def init() -> tuple[Artist]:
        """Initialize animation."""
        line.set_data([], [])
        return (line,)

    def animate(frame: int) -> tuple[Artist]:
        """Update animation frame."""
        start_idx = frame
        end_idx = frame + config.window_size

        if end_idx > len(signal.data):
            return (line,)

        window_data = signal.data[start_idx:end_idx]
        if update_func is not None:
            window_data = update_func(window_data)

        window_t = t[start_idx:end_idx]
        line.set_data(window_t, window_data)
        return (line,)

    return FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(signal.data) - config.window_size,
        interval=1000 / signal.sampling_rate,  # ms between frames
        blit=True,
    )


# %% Example Usage
if __name__ == "__main__":
    import doctest

    doctest.testmod()

    # Generate sample data
    ecg = generate_ecg(duration_s=10, sampling_rate=250)
    resp = generate_respiratory(duration_s=10, sampling_rate=50)

    # Static plots
    ecg_config = PlotConfig(title="ECG with Arrhythmia", color="red", window_size=1000)
    resp_config = PlotConfig(
        title="Respiratory Signal with Apnea", color="green", window_size=200
    )

    plot_signal(ecg, ecg_config)
    plot_signal(resp, resp_config)
    plt.show()

    # Animate with real-time filtering
    def apply_filter(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Apply bandpass filter to ECG data window.

        Example:
            >>> d = np.random.randn(100)
            >>> filtered = apply_filter(d)
            >>> len(filtered) == len(d)
            True

        """
        return preprocess_signal(
            Signal1D(data=data, sampling_rate=250, signal_type="ECG")
        )

    anim = animate_signal(ecg, ecg_config, update_func=apply_filter)
    plt.show()

    # Create and compile model
    model = create_classifier(input_length=2500, n_classes=2)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Assuming you have your data in X and y (features and labels respectively)
    # Example data generation for demonstration purposes
    X = np.random.rand(
        1000, 2500, 1
    )  # 1000 samples, each of length 2500 with a single channel
    y = np.random.randint(0, 2, size=1000)  # Binary labels

    # Step 3: Train the Model
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

    # Step 4: Evaluate the Model
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {accuracy}")

    # Step 5: Save the Model
    model.save("my_cnn_model.h5")

    # Step 6: Load the Model for Inference
    loaded_model = load_model("my_cnn_model.h5")

    # Step 7: Perform Inference
    new_data = np.random.rand(1, 2500, 1)  # New data point for inference
    predictions = loaded_model.predict(new_data)
    predicted_class = np.argmax(predictions, axis=1)[0]
    print(f"Predicted Class: {predicted_class}")
