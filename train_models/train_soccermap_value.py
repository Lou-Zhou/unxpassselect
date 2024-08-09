#train soccermap value
from pathlib import Path
from functools import partial

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import mlflow
from scipy.ndimage import zoom
from unxpass.components.utils import log_model, load_model
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
from unxpass.databases import SQLiteDatabase
from unxpass.datasets_custom import PassesDataset, CompletedPassesDataset, FailedPassesDataset
from unxpass.components import pass_selection, pass_value, pass_success
pd.options.mode.chained_assignment = None

DATA_DIR = Path("../stores/")
db = SQLiteDatabase(DATA_DIR / "database.sql")
dataset_success = partial(CompletedPassesDataset, DATA_DIR / "datasets" / "euro2020" / "train")
dataset_fail = partial(FailedPassesDataset, DATA_DIR / "datasets" / "euro2020" / "train")
value_fail = pass_value.SoccerMapComponent(model = pass_value.PytorchSoccerMapModel())
print("training...")
value_fail.train(dataset_fail, trainer = {"accelerator": "cpu", "devices":1, "max_epochs": 10})
mlflow.set_experiment("pass_value/soccermap")
with mlflow.start_run() as run:
    # Log the model
    mlflow.pytorch.log_model(value_fail.model, "model")

    # Retrieve the run ID
    run_id = run.info.run_id
    fail = run_id
    print(f"Fail Model saved with run_id: {run_id}")

value_success = pass_value.SoccerMapComponent(model = pass_value.PytorchSoccerMapModel())

value_success.train(dataset_success, trainer = {"accelerator": "cpu", "devices":1, "max_epochs": 10})
mlflow.set_experiment("pass_value/soccermap")
with mlflow.start_run() as run:
    # Log the model
    mlflow.pytorch.log_model(value_success.model, "model")

    # Retrieve the run ID
    run_id = run.info.run_id
    success = run_id
    print(f"Success Model saved with run_id: {run_id}")





print(f"Fail Model saved as {fail}")
print(f"Success Model saved as {success}")

#success: f94362f83b6c4f2caa5da826daaacb8d
#fail: f6f93a33c3ac4bc6a3b29442c0908dc7
"""
Can then load with:
model_pass_selection = pass_value.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        'run_id', map_location='cpu'
    )
)"""
