#train soccermap selection
"""
Originally ran from command line as: 
unxpass train \
  $(pwd)/config \
  $(pwd)/stores/datasets/euro2020/train \
  experiment="pass_selection/soccermap"
"""
from pathlib import Path
from functools import partial

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import mlflow
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
from unxpass.databases import SQLiteDatabase
from unxpass.datasets_custom import PassesDataset, CompletedPassesDataset, FailedPassesDataset
from unxpass.components import pass_selection, pass_value, pass_success
pd.options.mode.chained_assignment = None

DATA_DIR = Path("../stores/")
db = SQLiteDatabase(DATA_DIR / "database.sql")

dataset_train = partial(PassesDataset, DATA_DIR / "datasets" / "euro2020" / "train")

pass_selection_model = pass_selection.SoccerMapComponent(model = pass_selection.PytorchSoccerMapModel())
print("training...")
pass_selection_model.train(dataset_train, trainer = {"accelerator": "cpu", "devices":1, "max_epochs": 10})
#can test with 
#dataset_train = partial(PassesDataset, DATA_DIR / "datasets" / "euro2020" / "test")
#pass_selection_model.test(dataset_test)
mlflow.set_experiment("pass_value/soccermap")
with mlflow.start_run() as run:
    # Log the model
    mlflow.pytorch.log_model(pass_selection_model.model, "model")

    # Retrieve the run ID
    run_id = run.info.run_id
    fail = run_id
    print(f"Selection Model saved with run_id: {run_id}")

"""
Can then load with:
model_pass_selection = pass_selection.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        'run_id', map_location='cpu'
    )
)"""