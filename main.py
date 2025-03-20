import numpy as np
import pandas as pd
import torch
import timesfm
from chronos import BaseChronosPipeline
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

import os
import sys
# Add the diffusion_model folder to sys.path
sys.path.append(os.path.abspath("./diffusion_model"))
from diffusion_model.main import run_diffusion_model

horizon_length = 24

def load_dataset():

    # Load the dataset
    file_path = "datasets/Room Temperature/MLTempDataset1.csv"
    df = pd.read_csv(file_path)

    # Transform the dataset

    # Remove the 'Datetime1' column, as it's not needed
    #df = df.drop(columns=['Datetime1'])

    # Rename the unnamed column (e.g., 'Unnamed: 0') to 'unique_id'
    df = df.rename(columns={'Unnamed: 0': 'unique_id'})

    # Rename the columns to match the desired output format
    df = df.rename(columns={'Datetime': 'ds', 'Hourly_Temp': 'y'})

    # Convert 'ds' to datetime format
    df['ds'] = pd.to_datetime(df['ds'])

    # Reorder the columns to match the required format
    #df = df[['unique_id', 'ds', 'y']]

    return df


def run_timesfm(df):

    # Initialize the model and load a checkpoint

    tfm = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend="cpu",
          per_core_batch_size=32,
          horizon_len=horizon_length,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
    )

    """ tfm = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
        context_len=168,
        horizon_len=24,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
        backend="cpu",
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-1.0-200m"),
    ) """

    """ tfm = timesfm.TimesFm(
        context_len=168,
        horizon_len=24,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
        backend="cpu",
    )
    tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m") """

    """ tfm = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend="cpu",
          per_core_batch_size=32,
          context_len=168,
          horizon_len=24,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-1.0-200m"),
  ) """


    # Load base class

    """ Note that the four parameters are fixed to load the 200m model """
    """ input_patch_len=32
    output_patch_len=128
    num_layers=20
    model_dims=1280

    tcm = TimesFMConfig(input_patch_len, output_patch_len, num_layers, model_dims)
    tfm_torch = PatchedTimeSeriesDecoder(tcm)
    tfm_torch.load_state_dict(torch.load(PATH)) """

    """ model handles a max context length of 512 """
    """ tfm.forecast(<the input time series contexts>, <frequency>) """

    """ 
        frequency 0 -> up to daily granularity (T, MIN, H, D, B, U)
        frequency 1 -> weekly and monthly granularity (W, M)
        frequency 2 -> anything beyond monthly granularity, e.g. quarterly or yearly (Q, Y) 
    """
    
    
    # Forecast

    # numpy
    """ forecast_input = [
        np.sin(np.linspace(0, 20, 100)),
        np.sin(np.linspace(0, 20, 200)),
        np.sin(np.linspace(0, 20, 400)),
    ]
    frequency_input = [0, 1, 2]
    point_forecast, experimental_quantile_forecast = tfm.forecast(
        forecast_input,
        freq=frequency_input,
    ) """

    # pandas
    forecast_df = tfm.forecast_on_df(
        inputs=df,
        freq="h",  # hourly
        value_name="y",
        num_jobs=-1,
    )

    forecast = forecast_df.tail(24)
    forecast = forecast.iloc[:, :3]
    #forecast.to_csv('forecast_output.txt', index=False, sep='\t')

    return forecast

def run_chronos(df):
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

    
def plot_forecast(df, forecast_df):
    # Create a plot
    ax = df.plot(x=df.columns[1], y=df.columns[2], label="Original Data", color="blue")

    # Plot the forecast on the same axes
    forecast_df.plot(x=forecast_df.columns[1], y=forecast_df.columns[2], 
                    label="Forecast", color="red", ax=ax)

    # Formatting
    plt.xlabel("Datetime")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.title("Original Data vs Forecast")

    # Show the plot
    plt.show()

def evaluate_error(real_data, forecast_data):
    # Get 'y' values
    y_real_data = real_data["y"].to_numpy()
    y_forecast_data = forecast_data["y"].to_numpy()

    # Compute R²
    """
        R² = 1 → The model perfectly predicts the target.
        R² = 0 → The model is as good as predicting the mean of the target.
        R² < 0 → The model performs worse than just predicting the mean.
    """
    r2 = r2_score(y_real_data, y_forecast_data)
    print("\nR² Score:", r2)


def main():
    """ df = load_dataset()

    # Get a piece of the dataframe (week) 6864
    reduced_df = df.iloc[6696:6888].reset_index(drop=True)

    # Get the context (take back the last day that will be forecasted)
    context_df = reduced_df.iloc[:-24]

    #forecast_df = run_timesfm(context_df)
    forecast_df = run_chronos(context_df)

    plot_forecast(reduced_df, forecast_df)

    evaluate_error(reduced_df.tail(24), forecast_df) """

    """ args = {
        "gpu": 0,
        "config_path": "./Config/sines.yaml",
        "save_dir": "./toy_exp",
        "mode": "infill",
        "missing_ratio": 0.5,
        "milestone": 10
    }
    
    run_diffusion_model(args)  # Pass the dictionary as an argument """

    run_diffusion_model()


if __name__ == "__main__":
    main()
