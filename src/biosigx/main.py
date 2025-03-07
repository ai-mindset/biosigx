"""Main execution module for time series anomaly detection.

This module coordinates the overall anomaly detection pipeline, from data loading
and preprocessing to model training and evaluation. It serves as the main entry
point for the application.
"""

# %% [markdown]
# ## Anomaly Detection Pipeline
#
# This script orchestrates the entire anomaly detection process:
#
# 1. Load and preprocess time series data
# 2. Build and train an autoencoder model
# 3. Establish an anomaly detection threshold
# 4. Detect anomalies in new data
# 5. Visualise results
#
# This implementation
# - Uses minimal dependencies
# - Is fully typed and passes Pyright checks
# - Follows clean code principles with small, focused functions
# - Uses composition over inheritance
# - Handles data processing, model training, and anomaly detection in separate modules
# - Includes proper error handling and validation
# - Is easy to test and maintain

# %%
import keras
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from biosigx.data import load_data, process_data
from biosigx.logger import setup_logger
from biosigx.model import build_model
from biosigx.train import detect_anomalies, train_model
from biosigx.visualise import (
    plot_reconstruction_error,
    plot_time_series,
    plot_training_history,
    visualise_reconstruction,
)

# %% [markdown]
# ## Logger Setup

# %%
logger = setup_logger()


# %%
def run_anomaly_detection(
    data: str,
    sequence_length: int = 288,
    test_size: float = 0.2,
    batch_size: int = 128,
    epochs: int = 50,
    visualise: bool = True,
) -> tuple[NDArray[np.bool_], keras.Model, float]:
    r"""Run the complete anomaly detection pipeline.

    Args:
        data: CSV-formatted string containing time series data
        sequence_length: Length of each sequence
        test_size: Fraction of data to use for testing
        batch_size: Training batch size
        epochs: Maximum number of epochs
        visualise: Whether to generate visualizations

    Returns:
        Tuple of (anomaly flags, trained model, threshold)

    >>> import numpy as np
    >>> # Generate simple sine wave data
    >>> t = np.linspace(0, 20, 1000)
    >>> data = np.sin(t).reshape(-1, 1)
    >>> # Create CSV string
    >>> data_str = "\\n".join([str(x[0]) for x in data])
    >>> # Run detection (minimal epochs for test)
    >>> anomalies, model, threshold = run_anomaly_detection(
    ...     data_str, sequence_length=50, epochs=1, visualize=False
    ... )
    >>> isinstance(anomalies, np.ndarray)
    True

    """
    # Load data
    logger.info("Loading data...")
    df = load_data(data)

    # Process data into sequences
    logger.info("Processing data...")
    sequences, scaler = process_data(df, sequence_length)

    # Split into train and test sets
    logger.info(
        f"Splitting data into train/test sets ({1 - test_size:.0%}/{test_size:.0%})..."
    )
    train_sequences, test_sequences = train_test_split(
        sequences, test_size=test_size, shuffle=False
    )

    # Build model
    logger.info("Building model...")
    model = build_model(sequence_length)

    # Train model
    logger.info("Training model...")
    model, threshold = train_model(
        model, train_sequences, batch_size=batch_size, epochs=epochs
    )

    # Detect anomalies
    logger.info("Detecting anomalies...")
    train_predictions = model.predict(train_sequences, verbose=0)
    test_predictions = model.predict(test_sequences, verbose=0)

    train_errors = np.mean(np.abs(train_predictions - train_sequences), axis=(1, 2))
    test_errors = np.mean(np.abs(test_predictions - test_sequences), axis=(1, 2))

    train_anomalies = train_errors > threshold
    test_anomalies = test_errors > threshold

    # Combine train and test results
    all_errors = np.concatenate([train_errors, test_errors])
    all_anomalies = np.concatenate([train_anomalies, test_anomalies])

    # Print summary
    logger.info(
        f"Detected {np.sum(all_anomalies)} anomalies out of {len(all_anomalies)} sequences"
    )
    logger.info(f"Anomaly rate: {np.mean(all_anomalies):.2%}")

    # Visualise if requested
    if visualise:
        # Plot error distribution
        plot_reconstruction_error(all_errors, threshold)

        # Pass sequence_length to help handle sequence anomalies
        plot_time_series(
            df,
            all_anomalies,
            sequence_length=sequence_length,
            title="Time Series with Detected Anomalies",
        )

        # Visualise reconstructions
        normal_idx = np.where(~all_anomalies)[0][0]
        anomaly_idx = np.where(all_anomalies)[0][0] if np.any(all_anomalies) else None

        # Show normal reconstruction
        normal_original = sequences[normal_idx : normal_idx + 1]
        normal_reconstructed = model.predict(normal_original, verbose=0)
        visualise_reconstruction(
            normal_original, normal_reconstructed, index=0, figsize=(12, 6)
        )

        # Show anomalous reconstruction if anomalies were found
        if anomaly_idx is not None:
            anomaly_original = sequences[anomaly_idx : anomaly_idx + 1]
            anomaly_reconstructed = model.predict(anomaly_original, verbose=0)
            visualise_reconstruction(
                anomaly_original, anomaly_reconstructed, index=0, figsize=(12, 6)
            )

    return all_anomalies, model, threshold


# %% [markdown]
# ## Main Execution Point
#
# This section runs when the script is executed directly.

# %%
if __name__ == "__main__":
    import os
    import sys
    import tempfile

    # Generate synthetic data (sine wave with anomalies)
    logger.info("Generating synthetic data...")
    t = np.linspace(0, 50, 5000)
    values = np.sin(t) + 0.1 * np.random.randn(len(t))

    # Inject anomalies
    anomaly_indices = [1000, 2000, 3000, 4000]
    for idx in anomaly_indices:
        values[idx : idx + 50] += 2.0

    # Convert to CSV string
    data = "\n".join([str(x) for x in values])

    # Create a temporary file and write the data to it
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as temp_file:
        logger.info(f"Writing synthetic data to {temp_file.name}")
        temp_file.write(data)

    # Run anomaly detection
    anomalies, model, threshold = run_anomaly_detection(
        temp_file.name,
        sequence_length=100,  # Shorter sequence for demo
        epochs=20,
    )

    # Optionally, you can delete the temporary file if it's no longer needed
    os.remove(temp_file.name)

    logger.info("Anomaly detection completed successfully!")
