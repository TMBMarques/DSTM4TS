from chronos import BaseChronosPipeline
import torch
import numpy as np
import pandas as pd

def run_chronos_bolt(df, horizon_length):
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

    """ with open("forecast_output.txt", "w") as f:
        f.write(str(forecast.tolist())) """

    # Retrieve only the median forecast
    forecast_array = np.array(forecast)  # Convert to NumPy array
    median_forecast = forecast_array[0, 4, :]  # 0: first series, 4: median quantile (0.5)

    # Adjust the structure of the forecast (add IDs and timestamps)
    
    # Get last timestamp from existing DataFrame
    last_timestamp = df["ds"].iloc[-1]

    # Define new start date (next hour)
    start_date = last_timestamp + pd.Timedelta(hours=1)

    # Generate hourly timestamps
    date_range = pd.date_range(start=start_date, periods=len(median_forecast), freq='h')

    # Create unique IDs (e.g., "id_1", "id_2", ...)
    unique_ids = [f"id_{i+1}" for i in range(len(median_forecast))]

    # Create DataFrame
    forecast_df = pd.DataFrame({
        "unique_id": unique_ids,
        "ds": date_range,
        "y": median_forecast
    })

    return forecast_df
