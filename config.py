# config.py

from enum import Enum
from pre_trained_models.chronos_bolt import *
from pre_trained_models.time_moe import *
from pre_trained_models.timer import *

""" ------- EXECUTION ------- """
TRAIN = True
""" ------------------------- """


""" -------- DATASET -------- """
class Dataset(Enum):
    TEMPERATURE = ("Room Temperature", "./datasets/room_temperature.csv")   # Dimension: Hour
    PRICES = ("Aluminium Prices", "./datasets/aluminium_prices.csv")        # Dimension: Day
    PRODUCTION = ("Lemon Production", "./datasets/lemon_production.csv")    # Dimension: Month
    SINEWAVE = ("Sinewave", "./datasets/sinewave.csv")

    def __init__(self, label, file_path):
        self.label = label
        self.file_path = file_path

DATASET = Dataset.PRICES
SPLIT_TRAIN_TEST_INDEX = 1660
""" ------------------------- """


""" --- PRE-TRAINED MODEL --- """
class PreTrainedModel(Enum):
    CHRONOS_BOLT = ("Chronos-Bolt", "chronos", run_chronos_bolt, run_chronos_bolt_in_diffusion_model)
    TIME_MOE = ("Time-MoE", "timemoe", run_time_moe, run_time_moe_in_diffusion_model)
    TIMER = ("Timer", "timer", run_timer, run_timer_in_diffusion_model)

    def __init__(self, label, label_in_diffusion_model, forecast_fn, forecast_fn_in_diffusion_model):
        self.label = label
        self.label_in_diffusion_model = label_in_diffusion_model
        self.forecast_fn = forecast_fn
        self.forecast_fn_in_diffusion_model = forecast_fn_in_diffusion_model

    def run_forecast(self, df, datetime_column, horizon_length):
        return self.forecast_fn(df, datetime_column, horizon_length)
    
    def run_forecast_in_diffusion_model(self, df, horizon_length):
        return self.forecast_fn_in_diffusion_model(df, horizon_length)

PRE_TRAINED_MODEL = PreTrainedModel.TIMER
""" ------------------------- """


""" --------- MODEL --------- """
HORIZON_LENGTH = 100
DIFFUSION_MODEL_WINDOW = 100
DIFFUSION_MODEL_CONTEXT = 50
STRESS_WEIGHT = 1
TRAINING_MAX_EPOCHS = 30
BATCH_SIZE = 64

if DATASET == Dataset.TEMPERATURE:
    DIFFUSION_MODEL_CONFIG_YAML = "./diffusion_model/Config/room_temperature.yaml"
elif DATASET == Dataset.PRICES:
    DIFFUSION_MODEL_CONFIG_YAML = "./diffusion_model/Config/aluminium_prices.yaml"
elif DATASET == Dataset.PRODUCTION:
    DIFFUSION_MODEL_CONFIG_YAML = "./diffusion_model/Config/lemon_production.yaml"
elif DATASET == Dataset.SINEWAVE:
    DIFFUSION_MODEL_CONFIG_YAML = "./diffusion_model/Config/sinewave.yaml"
""" ------------------------- """
