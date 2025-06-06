from config import *
from auxiliar import *
from evaluation_metrics import print_realism_metrics, print_forecast_errors

import matplotlib.pyplot as plt

import torch
from transformers import AutoModelForCausalLM



def plot_forecast(df, forecast_df):
    # Create a plot
    
    plt.plot(df["ds"], df["y"], label="Original Data", color="blue", linestyle="-")
    plt.plot(forecast_df["ds"], forecast_df["y"], label="Original Data", color="red", linestyle="-")

    ax = plt.gca()

    # Formatting
    plt.xlabel("Datetime")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.title("Original Data vs Forecast")

    # Show the plot
    plt.show()

def main():
    # Load dataset
    df = load_dataset(DATASET.file_path)

    df_1 = df[5000:5100]

    # --- PRE_TRAINED MODEL ---

    model = AutoModelForCausalLM.from_pretrained(
        'Maple728/TimeMoE-50M',
        device_map="cpu",  # use "cpu" for CPU inference, and "cuda" for GPU inference.
        trust_remote_code=True,
    )

    # normalize seqs
    """ mean, std = seqs.mean(dim=-1, keepdim=True), seqs.std(dim=-1, keepdim=True)
    normed_seqs = (seqs - mean) / std """

    x = torch.tensor(df_1['y'].values, dtype=torch.float32).unsqueeze(0)

    


    # forecast
    prediction_length = HORIZON_LENGTH
    output = model.generate(x, max_new_tokens=prediction_length)  # shape is [batch_size, 12 + 6]
    normed_predictions = output[:, -prediction_length:]  # shape is [batch_size, 6]


    
   

    df_2 = df[5100:5200]
    df_2['y'] = normed_predictions.squeeze(0)

    plot_forecast(df[5000:5200], df_2)

    #-----------------------------------




if __name__ == "__main__":
    main()
