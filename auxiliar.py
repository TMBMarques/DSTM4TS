from ruamel.yaml import YAML
import pandas as pd
import matplotlib.pyplot as plt
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

    # Generate x-axis indices (0 to 199)
    x_df = list(range(len(df)))
    x_modified = list(range(DIFFUSION_MODEL_CONTEXT, DIFFUSION_MODEL_CONTEXT + len(modified_df)))
    x_forecast = list(range(len(df) - HORIZON_LENGTH, len(df)))
    x_modified_forecast = list(range(x_modified[-1] + 1, x_modified[-1] + 1 + len(modified_forecast_df)))

    # Add vertical background lines
    for x in x_df:
        plt.axvline(x=x, color="#1c1c1c", linestyle="-", linewidth=0.5, alpha=0.5)

    # Add dashed vertical lines
    plt.axvline(x=DIFFUSION_MODEL_CONTEXT, color="#545454", linestyle="--", label="Modified Context Start")
    plt.axvline(x=x_forecast[0], color="#545454", linestyle="--", label="Forecast Start")

    # Plot connection line for modifications
    plt.plot(
        [x_df[DIFFUSION_MODEL_CONTEXT - 1], x_modified[0]],
        [df["y"].iloc[DIFFUSION_MODEL_CONTEXT - 1], modified_df["y"].iloc[0]],
        color="#707070", linewidth=1, alpha=0.5
    )

    # Plot connection line for original forecast
    plt.plot(
        [x_df[-HORIZON_LENGTH - 1], x_forecast[0]],
        [df["y"].iloc[-HORIZON_LENGTH - 1], forecast_df["y"].iloc[0]],
        color="#707070", linewidth=1, alpha=0.5
    )

    # Plot connection line for modified forecast
    plt.plot(
        [x_modified[-1], x_modified_forecast[0]],
        [modified_df["y"].iloc[-1], modified_forecast_df["y"].iloc[0]],
        color="#EC9F05", linewidth=1, alpha=0.5
    )

    # Plot original data
    plt.plot(x_df, df["y"], label="Original Data", color="#707070", linestyle="-")

    # Plot modified context
    plt.plot(x_modified, modified_df["y"], label="Modified Context", color="#EC9F05", linestyle="-")

    # Plot original forecast
    plt.plot(x_forecast, forecast_df["y"], label="Original Forecast", color="#1098F7", linestyle="-")

    # Plot modified forecast
    plt.plot(x_modified_forecast, modified_forecast_df["y"], label="Modified Forecast", color="#BF3100", linestyle="-")

    # Format x-axis as step indices
    plt.xticks(ticks=range(0, len(df) + 1, 20))
    plt.xlabel("t")
    plt.ylabel("y", rotation=0)

    # Titles and legend
    highlight = f"$\\bf{{{DATASET.label} \ in\ {SPLIT_TRAIN_TEST_INDEX} \ with\ a\ stress\ weight\ of\ {STRESS_WEIGHT} \ in\ {PRE_TRAINED_MODEL.label}}}$"
    main_title = f"Original Data vs Modified Context and Forecasts for {highlight}"
    plt.title(main_title, pad=15)
    plt.legend()
    plt.tight_layout()
    plt.show()



def create_sinewave_csv(file_name='sinewave.csv', total_rows=1000, repetitions_per_50=3):
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

