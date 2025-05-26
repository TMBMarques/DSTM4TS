import numpy as np
import torch
from scipy.stats import wasserstein_distance
from dtaidistance import dtw
from statsmodels.tsa.stattools import acf
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


""" ------------- """
""" REALISM SCORE """
""" ------------- """


# - Compare distribution -
# Best around 0
# Lower scores mean more similar distributions, therefore, more realism
def get_wasserstein(real_data, generated_data):
    return wasserstein_distance(real_data, generated_data)


# Dynamic Time Warping
# - Compare temporal form -
# Temporal form: similarity of the trajectory of the curve (even with shifts or deformations)
# Best around 0
# Lower scores mean more realism
def get_dtw(real_data, generated_data):
    return dtw.distance(real_data, generated_data)


# Autocorrelation
# - Compare temporal structure -
# Temporal structure: similarity of the repetition patterns of the serie (seasonality, noise, ...)
# Best around 0
# Lower scores mean more realism
def get_acf(real, generated, nlags=20):
    acf_real = acf(real, nlags=nlags)
    acf_gen = acf(generated, nlags=nlags)
    return np.linalg.norm(acf_real - acf_gen)


""" # Compares the mean of the inner distances (intra-set) between real and generated samples
# Best around 1
# > 1 means generated samples are more variable
# < 1 means generated samples are less variable
def get_dispersion_ratio(real_samples, generated_samples):
    real_dists = pdist(real_samples, metric="euclidean")
    gen_dists = pdist(generated_samples, metric="euclidean")
    
    dispersion_real = np.mean(real_dists)
    dispersion_gen = np.mean(gen_dists)

    return dispersion_gen / dispersion_real if dispersion_real != 0 else np.inf

# Compares the distance between real and generated samples with the dispersion of the real data
# Best around 1
# > 1 means generated samples are more different than the real thata
def compute_distance_ratio(real_samples, generated_samples):
    inter_dists = cdist(real_samples, generated_samples, metric="euclidean")
    real_dists = pdist(real_samples, metric="euclidean")

    mean_inter = np.mean(inter_dists)
    mean_real_dispersion = np.mean(real_dists)

    return mean_inter / mean_real_dispersion if mean_real_dispersion != 0 else np.inf """


""" ----------------------- """
""" CONTEXT-FID WITH TS2VEC """
""" ----------------------- """

""" # Initiate model
model = TS2Vec(
    input_dims=1,              # ou mais, se multivariado
    output_dims=320,
    hidden_dims=64,
    depth=10,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    lr=0.001,
    batch_size=16,
    max_train_length=200,      # opcional
    temporal_unit=0
)

# Train
train_data = original_context_df['y'].values.reshape(1, -1, 1)  # formato (n_instâncias, n_tempos, n_variáveis)
model.fit(train_data, n_epochs=10, verbose=True)

# Save model
model.save('ts2vec_trained_model.pth')

# Load model
model_loaded = TS2Vec(input_dims=1)
model_loaded.load('ts2vec_trained_model.pth')

# -- juntar estas duas partes --

# Generate representations
#test_data = original_context_df['y'].values.reshape(1, -1, 1)
#encoded_repr = model_loaded.encode(test_data, encoding_window='full_series')

# data_real: (N, T, D) — N séries reais
# data_fake: (N, T, D) — N séries geradas
#emb_real = model.encode(data_real, encoding_window='full_series')
#emb_fake = model.encode(data_fake, encoding_window='full_series')

# ------------------------------ """





""" ---------------- """
""" PREDICTIVE SCORE """
""" ---------------- """


# R² = 1 → The model perfectly predicts the target.
# R² = 0 → The model is as good as predicting the mean of the target.
# R² < 0 → The model performs worse than just predicting the mean.
def get_r2(real_data, forecast_data):    
    return r2_score(real_data, forecast_data)


# MSE - Mean Squared Error
# ADVANTAGE : strongly penalizes large errors
# DISADVANTAGE : sensitive to scale and outliers
def get_mse(real_data, forecast_data):
    return np.mean((real_data - forecast_data) ** 2)


# RMSE - Root Mean Squared Error
# ADVANTAGE : strongly penalizes large errors
# DISADVANTAGE : sensitive to outliers
def get_rmse(real_data, forecast_data):
    return np.sqrt(np.mean((real_data - forecast_data)**2))


# MAE - Mean Absolute Error
# ADVANTAGE : treats all errors equally, without over-penalizing outliers
# DISADVANTAGE : doesn’t emphasize large errors
def get_mae(real_data, forecast_data):
    return mean_absolute_error(real_data, forecast_data)
