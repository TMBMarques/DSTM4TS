from ruamel.yaml import YAML
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from config import *



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



def load_dataset(file_path):
    df = pd.read_csv(file_path)

    return df



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
    last_point_before_modifications = DIFFUSION_MODEL_CONTEXT - 1
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
    highlight = f"$\\bf{{{DATASET.label} \ with\ a\ stress\ weight\ of\ {STRESS_WEIGHT}}}$"
    main_title = f"Original Data vs Modified Context and Forecasts for {highlight}"
    subtitle = "(training maximum epochs = {} | batch size = {})\n".format(TRAINING_MAX_EPOCHS, BATCH_SIZE)

    plt.suptitle(main_title, fontsize=12)  # Título principal
    plt.title(subtitle, fontsize=9)        # Parte menor (abaixo do principal)
    plt.legend()

    # Show the plot
    plt.show()



def create_sinusoidal_wave_csv(file_name='sinewave.csv', total_rows=1000, repetitions_per_50=3):
    # Configuration
    timesteps = np.arange(total_rows)  # from 0 to 999
    period = 50 / repetitions_per_50   # How many timesteps per full sine cycle

    # Generate sinusoidal wave values
    y = np.sin(2 * np.pi * timesteps / period)  # sine wave with amplitude 1 and proper frequency

    # Create DataFrame
    df = pd.DataFrame({
        'ds': timesteps,
        'y': y
    })

    # Save to CSV
    df.to_csv(file_name, index=False)
    print(f'CSV saved as {file_name}')

