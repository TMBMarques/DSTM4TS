from chronos import BaseChronosPipeline
import torch
import numpy as np
import pandas as pd

def run_chronos_bolt(df, datetime_column, horizon_length):
    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-tiny",
        device_map="cpu",  # use "cpu" for CPU inference and "mps" for Apple Silicon
        torch_dtype=torch.bfloat16,
    )

    # context must be either a 1D tensor, a list of 1D tensors,
    # or a left-padded 2D tensor with batch as the first dimension
    # Chronos-Bolt models generate quantile forecasts, so forecast has shape
    # [num_series, num_quantiles, prediction_length].
    forecast = pipeline.predict(
        context=torch.tensor(df["y"]), prediction_length=horizon_length
    )

    # Retrieve only the median forecast
    forecast_array = np.array(forecast)  # Convert to NumPy array
    median_forecast = forecast_array[0, 4, :]  # 0: first series, 4: median quantile (0.5)

    # Create DataFrame
    forecast_df = pd.DataFrame({
        "ds": datetime_column,
        "y": median_forecast
    })

    return forecast_df

def run_chronos_bolt_in_diffusion_model(df, horizon_length):
    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-tiny",
        device_map="cpu",  # use "cpu" for CPU inference and "mps" for Apple Silicon
        torch_dtype=torch.bfloat16,
    )

    # context must be either a 1D tensor, a list of 1D tensors,
    # or a left-padded 2D tensor with batch as the first dimension
    # Chronos-Bolt models generate quantile forecasts, so forecast has shape
    # [num_series, num_quantiles, prediction_length].
    forecast = pipeline.predict(
        context=df, prediction_length=horizon_length
    )

    median_forecast = forecast[:, 4, :] # Get median quantile 0.5

    return median_forecast
