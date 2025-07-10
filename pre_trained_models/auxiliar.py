import torch
import torch.nn.functional as F
from config import PRE_TRAINED_MODEL, DIFFUSION_MODEL_CONTEXT, DIFFUSION_MODEL_WINDOW, HORIZON_LENGTH



def get_forecast_loss_in_diffusion_model(original_data, modified_data, horizon_data):
    # Get initial part from original_data and last part from modified_data
    initial_original_data = original_data[:, :DIFFUSION_MODEL_WINDOW - DIFFUSION_MODEL_CONTEXT]
    last_modified_data = modified_data[:, DIFFUSION_MODEL_WINDOW - DIFFUSION_MODEL_CONTEXT:]
    context_data = torch.cat([initial_original_data, last_modified_data], dim=1)

    # Do forecasts
    forecasts = PRE_TRAINED_MODEL.run_forecast_in_diffusion_model(context_data, HORIZON_LENGTH)

    # Calculate forecast loss
    assert forecasts.shape == horizon_data.shape, "Mismatch between forecasts and targets"
    forecast_loss = F.mse_loss(forecasts, horizon_data, reduction='mean')
    # Using RMSE instead of MSE
    forecast_loss = torch.sqrt(forecast_loss)

    return forecast_loss
