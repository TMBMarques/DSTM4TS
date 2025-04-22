import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

import os
import sys
# Add the diffusion_model folder to sys.path
sys.path.append(os.path.abspath("./diffusion_model"))
from diffusion_model.main import run_diffusion_model

from pre_trained_models.models import PreTrainedModel
from pre_trained_models.chronos_bolt import run_chronos_bolt

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
    
""" def plot_forecast(df, forecast_df):
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
    plt.show() """

def plot_forecast(df, modified_df, forecast_df, modified_forecast_df):
    plt.figure(figsize=(12, 6))

    # Plot original data
    plt.plot(df["ds"], df["y"], label="Original Data", color="#4a4a4a", linestyle="-")

    # Plot modified context
    plt.plot(modified_df["ds"], modified_df["y"], label="Modified Context", color="#EC9F05", linestyle="-")

    # Plot original forecast
    plt.plot(forecast_df["ds"], forecast_df["y"], label="Original Forecast", color="#1098F7", linestyle="-")

    # Plot modified forecast
    plt.plot(modified_forecast_df["ds"], modified_forecast_df["y"], label="Modified Forecast", color="#BF3100", linestyle="-")

    # Add vertical dashed lines
    plt.axvline(x=modified_df["ds"].iloc[0], color="#d4d4d4", linestyle="--", label="Modified Context Start")
    plt.axvline(x=forecast_df["ds"].iloc[0], color="#d4d4d4", linestyle="--", label="Forecast Start")

    # Formatting
    plt.xlabel("Datetime")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.title("Original Data vs Modified Context and Forecasts")
    plt.legend()

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
    df = load_dataset()

    # Get a piece of the dataframe (week) 6864
    reduced_df = df.iloc[6240:6408].reset_index(drop=True)

    # Get the context (take back the last day that will be forecasted)
    original_context_df = reduced_df.iloc[:-24]

    # Run the diffusion model
    stress_weight = 0
    samples = run_diffusion_model(PreTrainedModel.CHRONOS_BOLT, stress_weight, original_context_df.copy())
    print(original_context_df.head())

    # Consolidate modified context (NOTE: this needs to be changed/improved)
    diffusion_model_window = 72
    samples = [sample[0] for sample in samples[diffusion_model_window]] # Pick the corresponding window (4th day, modifications happen on 5th and 6th)
    modified_context_df = original_context_df.iloc[96:].reset_index(drop=True) # Pick data from the 5th day, where modifications start
    modified_context_df['y'] = samples[24:] # Ignore the first 24 hours (from 4th day, to only pick the 5th and 6th)

    # Get predictions
    original_forecast_df = run_chronos_bolt(original_context_df, horizon_length)
    modified_forecast_df = run_chronos_bolt(modified_context_df, horizon_length)

    # Plot
    plot_forecast(reduced_df, modified_context_df, original_forecast_df, modified_forecast_df)

if __name__ == "__main__":
    main()
