import pandas as pd
from config import Dataset

def load_temperature():
    # Load the dataset
    file_path = "datasets/Room Temperature/MLTempDataset1.csv"
    df = pd.read_csv(file_path)

    # Transform the dataset

    """ # Rename the unnamed column (e.g., 'Unnamed: 0') to 'unique_id'
    df = df.rename(columns={'Unnamed: 0': 'unique_id'}) """

    # Drop unnamed column 'Unnamed: 0'
    df.drop(columns=['Unnamed: 0'], inplace=True)

    # Rename the columns to match the desired output format
    df = df.rename(columns={'Datetime': 'ds', 'Hourly_Temp': 'y'})

    # Convert 'ds' to datetime format
    df['ds'] = pd.to_datetime(df['ds'])

    return df

def load_prices():
    # Load the dataset
    file_path = "datasets/Aluminium Prices/Aluminium Historical Data_2012.csv"
    df = pd.read_csv(file_path)

    # Transform the dataset

    # Drop unnecessary columns
    df = df[["Date", "Price"]]

    # Rename the columns to match the desired output format
    df = df.rename(columns={'Date': 'ds', 'Price': 'y'})

    # Convert 'ds' to datetime format
    df['ds'] = pd.to_datetime(df['ds'])

    # Sort dates
    df = df.sort_values('ds').reset_index(drop=True)

    # Convert strings to float in column 'y'
    df["y"] = df["y"].str.replace(",", "", regex=False).astype(float)

    return df

def load_production():
    # Load the dataset
    file_path = "datasets/Lemon Production/Lemon.csv"
    df = pd.read_csv(file_path)

    # Transform the dataset

    # Merge Year and Month columns into datetime format, renaming it to 'ds'
    df["ds"] = pd.to_datetime(df[["Year", "Month"]].assign(Day=1))

    # Rename column to match the desired output format
    df = df.rename(column={'Price': 'y'})

    return df

LOADERS = {
    Dataset.TEMPERATURE: load_temperature,
    Dataset.PRICES: load_prices,
    Dataset.PRODUCTION: load_production,
}

def load_dataset(dataset: Dataset):
    try:
        return LOADERS[dataset]()
    except KeyError:
        raise ValueError(f"Dataset inválido: {dataset}")
    