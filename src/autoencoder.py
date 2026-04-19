import numpy as np
from sklearn.neural_network import MLPRegressor
import joblib

def build_and_train_autoencoder(X_normal):
    autoencoder = MLPRegressor(
        hidden_layer_sizes=(16, 8, 4, 8, 16),
        activation='relu',
        max_iter=200,
        random_state=42,
        verbose=False
    )
    autoencoder.fit(X_normal, X_normal)
    return autoencoder