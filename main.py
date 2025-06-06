import os
import sys
import numpy as np

sys.path.append(os.path.abspath("./diffusion_model")) # Add the diffusion_model folder to sys.path
from diffusion_model.main import run_diffusion_model

from config import *
from auxiliar import *
from evaluation_metrics import print_realism_metrics, print_forecast_errors

    

def main():
    # Load dataset
    df = load_dataset(DATASET.file_path)

    # Get an experimental window of the dataframe
    df = df.iloc[SPLIT_TRAIN_TEST_INDEX : SPLIT_TRAIN_TEST_INDEX + DIFFUSION_MODEL_WINDOW + HORIZON_LENGTH].reset_index(drop=True)

    # Get the context (remove the data that will be forecasted)
    original_context_df = df.iloc[:-HORIZON_LENGTH]

    # Get the real data that will be forecasted (horizon)
    horizon_df = df.tail(HORIZON_LENGTH).reset_index(drop=True)

    # Update diffusion model yaml config file
    update_diffusion_model_configuration()

    # Train or run the model

    # Train model
    if TRAIN:
        run_diffusion_model()
        
    # Run model
    else:
        samples = run_diffusion_model()

        # Consolidate modified context
        samples = samples.flatten()
        modified_context_df = original_context_df.iloc[DIFFUSION_MODEL_CONTEXT:].reset_index(drop=True)
        modified_context_df['y'] = samples[DIFFUSION_MODEL_CONTEXT:]

        # Get predictions
        original_forecast_df = PRE_TRAINED_MODEL.run_forecast(original_context_df, horizon_df['ds'], HORIZON_LENGTH)
        modified_forecast_df = PRE_TRAINED_MODEL.run_forecast(modified_context_df, horizon_df['ds'], HORIZON_LENGTH)

        # Get realism metrics

        # Get the multiple data from the modified interval
        modified_interval_original_data = original_context_df.iloc[DIFFUSION_MODEL_CONTEXT:].reset_index(drop=True)['y'].to_numpy()
        modified_interval_modified_data = modified_context_df['y'].to_numpy()

        # Get an array with the mean value of the time series to serve as a baseline for realism comparison
        mean_value = original_context_df['y'].mean()
        modified_interval_mean_data = np.full(DIFFUSION_MODEL_WINDOW - DIFFUSION_MODEL_CONTEXT, mean_value)

        print_realism_metrics(modified_interval_original_data, modified_interval_modified_data, modified_interval_mean_data)

        # Get forecast errors

        # Get the multiple data from the forecasted interval
        horizon_data = horizon_df["y"].to_numpy()
        original_forecast_data = original_forecast_df["y"].to_numpy()
        modified_forecast_data = modified_forecast_df["y"].to_numpy()

        print_forecast_errors(horizon_data, original_forecast_data, modified_forecast_data)

        # Plot results
        plot_forecast(df, modified_context_df, original_forecast_df, modified_forecast_df)
    

if __name__ == "__main__":
    main()
