# config.py

from enum import Enum

""" ----- EXECUTION ----- """
TRAIN = False
""" --------------------- """


""" ------ DATASET ------ """
class Dataset(Enum):
    TEMPERATURE = "Room Temperature"     # Dimension: Hour
    PRICES = "Aluminium Prices"          # Dimension: Day
    PRODUCTION = "Lemon Production"      # Dimension: Month

DATASET = Dataset.TEMPERATURE

DATA_START = 2282#1023
DATA_END = 2450#1063
""" --------------------- """


""" ------- MODEL ------- """
HORIZON_LENGTH = 24#40#10 #24
DIFFUSION_MODEL_WINDOW = 72#60#20 #72
DIFFUSION_MODEL_CONTEXT = 24#40#10 #24
STRESS_WEIGHT = 30
TRAINING_MAX_EPOCHS = 5000
BATCH_SIZE = 64

if DATASET == Dataset.TEMPERATURE:
    DIFFUSION_MODEL_CONFIG_YAML = "./diffusion_model/Config/room_temperature.yaml"
elif DATASET == Dataset.PRICES:
    DIFFUSION_MODEL_CONFIG_YAML = "./diffusion_model/Config/aluminium_prices.yaml"
elif DATASET == Dataset.PRODUCTION:
    DIFFUSION_MODEL_CONFIG_YAML = "./diffusion_model/Config/lemon_production.yaml"
""" --------------------- """
