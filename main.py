import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from ruamel.yaml import YAML

import os
import sys

# Add the diffusion_model folder to sys.path
sys.path.append(os.path.abspath("./diffusion_model"))

from diffusion_model.main import run_diffusion_model

from pre_trained_models.models import PreTrainedModel
from pre_trained_models.chronos_bolt import run_chronos_bolt

from config import *
from datasets import load_dataset
from evaluation_metrics import *


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


def update_diffusion_model_configuration():
    yaml = YAML()
    yaml.preserve_quotes = True

    # Get data from yaml
    with open(DIFFUSION_MODEL_CONFIG_YAML, "r") as f:
        config = yaml.load(f)

    # Update yaml data
    config["model"]["params"]["seq_length"] = DIFFUSION_MODEL_WINDOW
    config["solver"]["max_epochs"] = TRAINING_MAX_EPOCHS
    config["solver"]["save_cycle"] = int(TRAINING_MAX_EPOCHS / 10)
    config["dataloader"]["train_dataset"]["params"]["window"] = DIFFUSION_MODEL_WINDOW
    config["dataloader"]["test_dataset"]["params"]["window"] = DIFFUSION_MODEL_WINDOW
    config["dataloader"]["batch_size"] = BATCH_SIZE

    # Save updates into the file
    with open(DIFFUSION_MODEL_CONFIG_YAML, "w") as f:
        yaml.dump(config, f)



def plot_forecast(df, modified_df, forecast_df, modified_forecast_df):
    plt.figure(figsize=(12, 6), facecolor='lightgray')
    ax = plt.gca()
    ax.set_facecolor('#242424')

    # Add vertical date lines
    for date in df["ds"]:
        plt.axvline(x=date, color="#1c1c1c", linestyle="-", linewidth=0.5, alpha=0.5)

    # Add vertical dashed lines
    plt.axvline(x=modified_df["ds"].iloc[0], color="#545454", linestyle="--", label="Modified Context Start")
    plt.axvline(x=forecast_df["ds"].iloc[0], color="#545454", linestyle="--", label="Forecast Start")

    # Plot conection line for modifications
    last_point_before_modifications = DATA_END - DATA_START - DIFFUSION_MODEL_WINDOW - 1
    plt.plot(
        [df["ds"].iloc[last_point_before_modifications], modified_df["ds"].iloc[0]],
        [df["y"].iloc[last_point_before_modifications], modified_df["y"].iloc[0]],
        color="#707070",
        linewidth=1,
        alpha=0.5
    )

    # Plot conection line for original forecast
    last_point_before_original_forecast = - HORIZON_LENGTH - 1
    plt.plot(
        [df["ds"].iloc[last_point_before_original_forecast], forecast_df["ds"].iloc[0]],
        [df["y"].iloc[last_point_before_original_forecast], forecast_df["y"].iloc[0]],
        color="#707070",
        linewidth=1,
        alpha=0.5
    )

    # Plot conection line for original forecast
    plt.plot(
        [modified_df["ds"].iloc[-1], modified_forecast_df["ds"].iloc[0]],
        [modified_df["y"].iloc[-1], modified_forecast_df["y"].iloc[0]],
        color="#EC9F05",
        linewidth=1,
        alpha=0.5
    )

    # Plot original data
    plt.plot(df["ds"], df["y"], label="Original Data", color="#707070", linestyle="-")
    plt.scatter(df["ds"], df["y"], color="#707070", s=5)

    # Plot modified context
    plt.plot(modified_df["ds"], modified_df["y"], label="Modified Context", color="#EC9F05", linestyle="-")
    plt.scatter(modified_df["ds"], modified_df["y"], color="#EC9F05", s=5)

    # Plot original forecast
    plt.plot(forecast_df["ds"], forecast_df["y"], label="Original Forecast", color="#1098F7", linestyle="-")
    plt.scatter(forecast_df["ds"], forecast_df["y"], color="#1098F7", s=5)

    # Plot modified forecast
    plt.plot(modified_forecast_df["ds"], modified_forecast_df["y"], label="Modified Forecast", color="#BF3100", linestyle="-")
    plt.scatter(modified_forecast_df["ds"], modified_forecast_df["y"], color="#BF3100", s=5)

    # Format x-axis with more detailed dates
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%Y"))
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0.015, -0.05, 1, 0.9])


    # Labels and legend
    plt.xlabel("Datetime")
    plt.ylabel("Value")
    plt.xticks(rotation=0)
    highlight = f"$\\bf{{{DATASET.value} \ with\ a\ stress\ weight\ of\ {STRESS_WEIGHT}}}$"
    main_title = f"Original Data vs Modified Context and Forecasts for {highlight}"
    subtitle = "(training maximum epochs = {} | batch size = {})\n".format(TRAINING_MAX_EPOCHS, BATCH_SIZE)

    plt.suptitle(main_title, fontsize=12)  # Título principal
    plt.title(subtitle, fontsize=9)        # Parte menor (abaixo do principal)
    plt.legend()

    # Show the plot
    plt.show()
    

def main():
    # Load dataset
    df = load_dataset(DATASET)

    # Get a piece of the dataframe
    reduced_df = df.iloc[DATA_START:DATA_END].reset_index(drop=True)

    # Get the context (remove the data that will be forecasted)
    original_context_df = reduced_df.iloc[:-HORIZON_LENGTH]

    # Get the real data that will be forecasted
    original_data = reduced_df.tail(HORIZON_LENGTH).reset_index(drop=True)

    # Get the piece of data that will be sent to the diffusion model
    diffusion_model_df = original_context_df.iloc[DATA_END - DATA_START - DIFFUSION_MODEL_WINDOW - DIFFUSION_MODEL_CONTEXT:].reset_index(drop=True)

    # Update diffusion model yaml config file
    update_diffusion_model_configuration()

    # Train or run the model

    # Train model
    if TRAIN:
        run_diffusion_model(df.copy())
        
    # Run model
    else:
        samples = run_diffusion_model(diffusion_model_df.copy())

        # Consolidate modified context
        samples = samples.flatten()
        modified_context_df = original_context_df.copy()
        modified_context_df.loc[modified_context_df.tail(DIFFUSION_MODEL_WINDOW - DIFFUSION_MODEL_CONTEXT).index, 'y'] = samples[DIFFUSION_MODEL_CONTEXT:]

        # Get predictions
        original_forecast_df = run_chronos_bolt(original_context_df, original_data['ds'], HORIZON_LENGTH)
        modified_forecast_df = run_chronos_bolt(modified_context_df, original_data['ds'], HORIZON_LENGTH)

        # Get realism metrics

        # Get y column for the modificated interval
        realism_y_real_data = diffusion_model_df.iloc[DIFFUSION_MODEL_CONTEXT:].reset_index(drop=True)['y'].to_numpy()
        modified_context_df = modified_context_df.tail(DIFFUSION_MODEL_WINDOW - DIFFUSION_MODEL_CONTEXT).reset_index(drop=True)
        realism_y_generated_data = modified_context_df["y"].to_numpy()

        # Get an array with the mean value of the time series to serve as a baseline for realism comparison
        mean_y = original_context_df['y'].mean()
        realism_mean_array = np.full(DIFFUSION_MODEL_WINDOW - DIFFUSION_MODEL_CONTEXT, mean_y)

        print("\n Realism Metrics \n")
        print(f"(Wasserstein) modifications: {get_wasserstein(realism_y_real_data, realism_y_generated_data)}\n")
        print(f"(Wasserstein) mean value: {get_wasserstein(realism_y_real_data, realism_mean_array)}\n")
        print(f"(DTW) modifications: {get_dtw(realism_y_real_data, realism_y_generated_data)}\n")
        print(f"(DTW) mean value: {get_dtw(realism_y_real_data, realism_mean_array)}\n")
        print(f"(ACF) modifications: {get_acf(realism_y_real_data, realism_y_generated_data)}\n")
        #print(f"(ACF) mean value: {get_acf(realism_y_real_data, realism_mean_array)}\n")

        # Get forecast errors

        # Get y column for the forecasted interval
        y_original_data = original_data["y"].to_numpy()
        y_original_forecast_data = original_forecast_df["y"].to_numpy()
        y_modified_forecast_data = modified_forecast_df["y"].to_numpy()

        print("\n Forecast Errors \n")
        print(f"(MSE) Original error: {get_mse(y_original_data, y_original_forecast_data)}\n")
        print(f"(MSE) Modified error: {get_mse(y_original_data, y_modified_forecast_data)}\n")
        print(f"(RMSE) Original error: {get_rmse(y_original_data, y_original_forecast_data)}\n")
        print(f"(RMSE) Modified error: {get_rmse(y_original_data, y_modified_forecast_data)}\n")
        print(f"(MAE) Original error: {get_mae(y_original_data, y_original_forecast_data)}\n")
        print(f"(MAE) Modified error: {get_mae(y_original_data, y_modified_forecast_data)}\n")

        # Plot
        plot_forecast(reduced_df, modified_context_df, original_forecast_df, modified_forecast_df)
    

if __name__ == "__main__":
    main()
