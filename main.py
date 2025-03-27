import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

import os
import sys
# Add the diffusion_model folder to sys.path
sys.path.append(os.path.abspath("./diffusion_model"))
from diffusion_model.main import run_diffusion_model

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
    df = load_dataset()

    # Get a piece of the dataframe (week) 6864
    reduced_df = df.iloc[6696:6888].reset_index(drop=True)

    # Get the context (take back the last day that will be forecasted)
    context_df = reduced_df.iloc[:-24]

    #forecast_df = run_timesfm(context_df, horizon_length)
    """ forecast_df = run_chronos_bolt(context_df, horizon_length)

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

    run_diffusion_model(context_df)


if __name__ == "__main__":
    main()
