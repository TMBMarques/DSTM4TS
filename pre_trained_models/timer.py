# Timer
# Hugging Face : https://huggingface.co/thuml/timer-base-84m



import torch
import pandas as pd
from transformers import AutoModelForCausalLM

def run_timer(df, datetime_column, horizon_length):
    # Load pre-trained model
    model = AutoModelForCausalLM.from_pretrained('thuml/timer-base-84m', trust_remote_code=True)

    # Prepare input
    seqs = torch.tensor(df['y'].values, dtype=torch.float32).unsqueeze(0)

    print(seqs.shape)

    # Generate forecast
    output = model.generate(seqs, max_new_tokens=horizon_length)

    # Create DataFrame
    forecast_df = pd.DataFrame({
        "ds": datetime_column,
        "y": output.squeeze(0)
    })

    return forecast_df



def run_timer_in_diffusion_model(df, horizon_length):
    # Load pre-trained model
    model = AutoModelForCausalLM.from_pretrained('thuml/timer-base-84m', trust_remote_code=True)

    # Generate forecast
    output = model.generate(df, max_new_tokens=horizon_length)

    return output
