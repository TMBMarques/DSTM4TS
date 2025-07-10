import numpy as np
from scipy.stats import wasserstein_distance
from dtaidistance import dtw
from statsmodels.tsa.stattools import acf
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error



""" ------------- """
""" REALISM SCORE """
""" ------------- """



# - Compare distribution -
# Best around 0
# Lower scores mean more similar distributions, therefore, more realism
def get_wasserstein(real_data, generated_data):
    return round(wasserstein_distance(real_data, generated_data), 2)



# Dynamic Time Warping
# - Compare temporal form -
# Temporal form: similarity of the trajectory of the curve (even with shifts or deformations)
# Best around 0
# Lower scores mean more realism
def get_dtw(real_data, generated_data):
    return round(dtw.distance(real_data, generated_data), 2)



# Autocorrelation Function
# - Compare temporal structure -
# Temporal structure: similarity of the repetition patterns of the serie (seasonality, noise, ...)
# Best around 0
# Lower scores mean more realism
def get_acf(real, generated, nlags=20):
    acf_real = acf(real, nlags=nlags)
    acf_gen = acf(generated, nlags=nlags)
    return round(np.linalg.norm(acf_real - acf_gen), 2)



def print_realism_metrics(real_data, modified_data, mean_data):
    print("\n Realism Metrics \n")
    print(f"(Wasserstein) modifications: {get_wasserstein(real_data, modified_data)}\n")
    print(f"(Wasserstein) mean value: {get_wasserstein(real_data, mean_data)}\n")
    print(f"(DTW) modifications: {get_dtw(real_data, modified_data)}\n")
    print(f"(DTW) mean value: {get_dtw(real_data, mean_data)}\n")
    print(f"(ACF) modifications: {get_acf(real_data, modified_data)}\n")
    #print(f"(ACF) mean value: {get_acf(real_data, mean_data)}\n")



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
    return round(np.mean((real_data - forecast_data) ** 2), 2)



# RMSE - Root Mean Squared Error
# ADVANTAGE : strongly penalizes large errors
# DISADVANTAGE : sensitive to outliers
def get_rmse(real_data, forecast_data):
    return round(np.sqrt(np.mean((real_data - forecast_data)**2)), 2)



# MAE - Mean Absolute Error
# ADVANTAGE : treats all errors equally, without over-penalizing outliers
# DISADVANTAGE : doesn’t emphasize large errors
def get_mae(real_data, forecast_data):
    return round(mean_absolute_error(real_data, forecast_data), 2)



def print_forecast_errors(original_data, original_forecast_data, modified_forecast_data):
    print("\n Forecast Errors \n")
    print(f"(MSE) Original error: {get_mse(original_data, original_forecast_data)}\n")
    print(f"(MSE) Modified error: {get_mse(original_data, modified_forecast_data)}\n")
    print(f"(RMSE) Original error: {get_rmse(original_data, original_forecast_data)}\n")
    print(f"(RMSE) Modified error: {get_rmse(original_data, modified_forecast_data)}\n")
    print(f"(MAE) Original error: {get_mae(original_data, original_forecast_data)}\n")
    print(f"(MAE) Modified error: {get_mae(original_data, modified_forecast_data)}\n")
    