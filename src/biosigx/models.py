"""Build and return an encoder-decoder autoencoder model for anomaly detection."""

# %%
from keras import Model, layers


# %%
def build_autoencoder(
    sequence_length: int, n_features: int, latent_dim: int = 32
) -> tuple[Model, Model, Model]:
    """Build encoder-decoder architecture for anomaly detection.

    Args:
        sequence_length: Length of input sequences
        n_features: Number of features per timestep
        latent_dim: Dimension of latent space

    Returns:
        autoencoder, encoder, decoder models

    """
    # Encoder
    encoder_inputs = layers.Input(shape=(sequence_length, n_features))
    x = layers.LSTM(64, return_sequences=True)(encoder_inputs)
    x = layers.LSTM(32, return_sequences=False)(x)
    encoder_outputs = layers.Dense(latent_dim)(x)
    encoder = Model(encoder_inputs, encoder_outputs, name="encoder")

    # Decoder
    decoder_inputs = layers.Input(shape=(latent_dim,))
    x = layers.RepeatVector(sequence_length)(decoder_inputs)
    x = layers.LSTM(32, return_sequences=True)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    decoder_outputs = layers.TimeDistributed(layers.Dense(n_features))(x)
    decoder = Model(decoder_inputs, decoder_outputs, name="decoder")

    # Autoencoder
    autoencoder_inputs = layers.Input(shape=(sequence_length, n_features))
    encoded = encoder(autoencoder_inputs)
    decoded = decoder(encoded)
    autoencoder = Model(autoencoder_inputs, decoded, name="autoencoder")

    return autoencoder, encoder, decoder
