import random
import warnings
warnings.filterwarnings('ignore')
import argparse
from pathlib import Path
import re
import os
import numpy as np
import pandas as pd
import string
import pickle
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForTokenClassification, AutoTokenizer, EarlyStoppingCallback
from ast import literal_eval
from DatasetLoader import DatasetLoader
from sklearn.model_selection import KFold
from utils import *
import torch
from const import *
from datetime import datetime, timedelta


# Path to your PKL file
# file_path = r"data/V2_train.pkl"
# file_path = r"data/V2_test.pkl"
file_path = r"data/V2_zero_shot.pkl"

# Load the PKL file
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Check if the data is a pandas DataFrame
if not isinstance(data, pd.DataFrame):
    raise ValueError("The loaded data is not a pandas DataFrame.")

# Generate random timestamps between 1996-01-01 and 2024-12-31
def generate_random_timestamps(start_year, end_year, num_samples):
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    random_dates = [
        start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
        for _ in range(num_samples)
    ]
    return random_dates

# Add a new 'timestamps' column to the DataFrame
data['timestamps'] = generate_random_timestamps(1996, 2024, len(data))

# Save the updated DataFrame back to the PKL file
with open(file_path, 'wb') as file:
    pickle.dump(data, file)

# Optional: Save to CSV for viewing in Excel
data.to_csv('output_with_timestamps_zero_shot.csv', index=False)
print("Updated data saved to 'output_with_timestamps.csv'")