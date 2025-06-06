# Time-MoE
# Hugging Face : https://huggingface.co/Maple728/TimeMoE-50M

import torch
import pandas as pd
from transformers import AutoModelForCausalLM

def run_time_moe(df, datetime_column, horizon_length):
    # Load pre-trained model
    model = AutoModelForCausalLM.from_pretrained(
        'Maple728/TimeMoE-50M',
        device_map="cpu",  # use "cpu" for CPU inference, and "cuda" for GPU inference.
        trust_remote_code=True,
    )

    # Prepare input
    seqs = torch.tensor(df['y'].values, dtype=torch.float32).unsqueeze(0)

    # Normalize input
    mean, std = seqs.mean(dim=1, keepdim=True), seqs.std(dim=1, keepdim=True)
    normed_seqs = (seqs - mean) / std

    # Generate forecast
    output = model.generate(normed_seqs, max_new_tokens=horizon_length)  # shape is [batch_size, 12 + 6]
    normed_predictions = output[:, -horizon_length:]  # shape is [batch_size, 6]

    # inverse normalize
    predictions = normed_predictions * std + mean

    # Create DataFrame
    forecast_df = pd.DataFrame({
        "ds": datetime_column,
        "y": predictions.squeeze(0)
    })

    return forecast_df

def run_time_moe_in_diffusion_model(df, horizon_length):
    # Load pre-trained model
    model = AutoModelForCausalLM.from_pretrained(
        'Maple728/TimeMoE-50M',
        device_map="cpu",  # use "cpu" for CPU inference, and "cuda" for GPU inference.
        trust_remote_code=True,
    )

    # Generate forecast
    output = model.generate(df, max_new_tokens=horizon_length)  # shape is [batch_size, 12 + 6]
    predictions = output[:, -horizon_length:]

    return predictions
