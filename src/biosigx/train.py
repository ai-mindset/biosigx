"""Train and return an autoencoder model for anomaly detection."""

# %%
import numpy as np
from keras import Model, callbacks

from biosigx.models import build_autoencoder


# %%
def train_anomaly_detector(
    X_train: np.ndarray,
    seq_length: int,
    n_features: int,
    epochs: int = 100,
    batch_size: int = 32,
    validation_split: float = 0.1,
    latent_dim: int = 32,
) -> tuple[Model, Model, Model]:
    """Train autoencoder for anomaly detection.

    Args:
        X_train: Training data
        X_test: Test data
        seq_length: Length of input sequences
        n_features: Number of features
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Proportion of training data to use for validation
        latent_dim: Dimension of latent space

    Returns:
        Trained autoencoder, encoder and decoder models

    """
    # Build model
    autoencoder, encoder, decoder = build_autoencoder(seq_length, n_features, latent_dim)

    # Compile
    autoencoder.compile(
        optimizer="adam",
        loss="mse",
    )

    # Train
    autoencoder.fit(
        X_train,
        X_train,  # Autoencoder reconstructs its input
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            ),
        ],
    )

    return autoencoder, encoder, decoder
