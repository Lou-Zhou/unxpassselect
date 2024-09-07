"""
Replace ratings.py with this for selection criteria
"""

from typing import Callable
from socceraction.spadl.utils import add_names
import numpy as np
import pandas as pd
import torch
import itertools
from rich.progress import track
from torch.utils.data import DataLoader, Subset, random_split
from unxpass.components import pass_selection, pass_success, pass_value, pass_value_custom
from unxpass.datasets_custom import PassesDataset
def convert_pass_coords(pass_selection_surface, x_t, y_t):
    #edited to convert selection surface coords to real x and y
    p_odds = pass_selection_surface[y_t, x_t]
    y_dim, x_dim = pass_selection_surface.shape
    y_t = y_t / y_dim * 68 + 68 / y_dim / 2
    x_t = x_t / x_dim * 105 + 105 / x_dim / 2
    
    return x_t, y_t, p_odds
def convert_x(x_t, x_dim):
    return x_t / x_dim * 105 + 105 / x_dim / 2
def convert_y(y_t, y_dim):
    return y_t / y_dim * 68 + 68 / y_dim / 2

class LocationPredictions:
    def __init__(
        self,
        pass_selection_component: pass_selection.SoccerMapComponent,
        pass_success_component: pass_success.SoccerMapComponent,
        pass_value_success_offensive_component: pass_value_custom.SoccerMapComponent,
        pass_value_success_defensive_component: pass_value.SoccerMapComponent,
        pass_value_fail_offensive_component: pass_value.SoccerMapComponent,
        pass_value_fail_defensive_component: pass_value.SoccerMapComponent
    ):
        self.pass_value_success_offensive_component = pass_value_success_offensive_component
        self.pass_value_success_defensive_component = pass_value_success_defensive_component
        self.pass_selection_component = pass_selection_component
        self.pass_success_component = pass_success_component
        self.pass_value_fail_offensive_component = pass_value_fail_offensive_component
        self.pass_value_fail_defensive_component = pass_value_fail_defensive_component
        
    def rate_all_games(self, db, dataset, summarize = True, custom_pass = None):
        print("Generating Surfaces:")
        pass_selection_surface = self.pass_selection_component.predict_surface(dataset, db = db)
        pass_success_surface = self.pass_success_component.predict_surface(dataset,db = db)
        pass_value_surface_offensive_success = self.pass_value_success_offensive_component.predict_surface(dataset, db = db)
        pass_value_surface_offensive_fail = self.pass_value_fail_offensive_component.predict_surface(dataset, db = db)
        pass_value_surface_defensive_success = self.pass_value_success_defensive_component.predict_surface(dataset, db = db)
        pass_value_surface_defensive_fail = self.pass_value_fail_defensive_component.predict_surface(dataset, db = db)
        print("Finished.")
        alldf = []
        sels = self.pass_selection_component.predict(dataset)#ensuring that the actual pass made is in the dataset
        value_fail = self.pass_value_fail_offensive_component.predict(dataset) - self.pass_value_fail_defensive_component.predict(dataset)
        value_success = self.pass_value_success_offensive_component.predict(dataset) - self.pass_value_success_defensive_component.predict(dataset)
        success = self.pass_success_component.predict(dataset)
        all_ratings = pd.concat([sels, success, value_fail, value_success], axis=1).rename(columns = {0:"selection_probability", 1: "success_probability", 2:"value_fail", 3:"value_success"})
        for game in pass_selection_surface:
            game_preds = all_ratings.loc[game]
            ending_coords = db.actions(game_id = game)[["end_x", "end_y"]].loc[game]
            game_ogs = pd.concat([game_preds,ending_coords], axis = 1).dropna()
            for action in pass_selection_surface[game]:
                if custom_pass is not None:
                    game = custom_pass["game_id"]
                    action = custom_pass["action_id"]
                print(game, action)
                true_ends = db.actions(game_id = game).loc[(game,action)][["end_x","end_y"]]
                
                metrics = self.rate(game, action, pass_selection_surface, pass_success_surface, pass_value_surface_offensive_success, pass_value_surface_defensive_success,pass_value_surface_offensive_fail, pass_value_surface_defensive_fail, db)
                metrics["Dist_From_True"] = (metrics["end_x"] - true_ends["end_x"])**2 + (metrics["end_y"] - true_ends["end_y"])**2
                closest_idx = np.where(metrics["Dist_From_True"] == min(metrics["Dist_From_True"]))[0]#getting closest pass to actual pass and then replacing that one with the original pass
                metrics = metrics.reset_index(drop = True)
                cols = ["end_x", "end_y", "selection_probability", "success_probability", "value_success", "value_fail"]
                metrics["True_Location"] = 0
                for col in cols:
                    metrics[col][closest_idx[0]] = game_ogs[col].loc[action]
                metrics["selection_probability"] = np.float64(metrics["selection_probability"])
                metrics["True_Location"][closest_idx[0]] = 1
                metrics["expected_utility"] = (metrics["success_probability"] * metrics["value_success"]) + ((1 - metrics["success_probability"]) * metrics["value_fail"])
                metrics["evaluation_criterion"] = metrics["expected_utility"] - sum(metrics["selection_probability"] * metrics["expected_utility"])
                metrics["selection_criterion"] = sum(
                metrics["selection_probability"] * (metrics["evaluation_criterion"])**2
                )
                
                metrics = metrics.drop(columns = ["Dist_From_True"])
                if summarize:
                    metrics = metrics[metrics["True_Location"] == 1]
                    metrics = metrics[["original_event_id", "game_id", "action_id", "start_x","start_y", "end_x", "end_y", "result_id", "selection_criterion", "evaluation_criterion"]]
                if custom_pass is not None:
                    return metrics
                alldf.append(metrics)
            
        combined = pd.concat(alldf)
        return combined
    def rate(self, game, action, selection, success, value_success_off, value_success_def, value_fail_off, value_fail_def, db):
        game_selection = selection[game][action]
        game_success = success[game][action]
        game_value_success_off = value_success_off[game][action]
        game_value_success_def = value_success_def[game][action]
        game_value_fail_off = value_fail_off[game][action]
        game_value_fail_def = value_fail_def[game][action]
        x_lim, y_lim = game_selection.shape
        coords = list(itertools.product(range(0,x_lim), range(0,y_lim)))
        test_db = db.actions(game_id = game)

        df_override = pd.DataFrame([test_db.loc[(game, action)]] * len(coords))
        df_override["game_id"] = game
        df_override["action_id"] = action
        df_override["coord_x"] = [i[0] for i in coords]
        df_override["coord_y"] = [i[1] for i in coords]
        df_override["end_x"] = convert_x(df_override["coord_x"], x_lim)
        df_override["end_y"] = convert_y(df_override["coord_y"], y_lim)
        df_override["selection_probability"] = game_selection[df_override["coord_x"], df_override["coord_y"]]
        df_override["success_probability"] = game_success[df_override["coord_x"], df_override["coord_y"]]
        df_override["value_success_off"] = game_value_success_off[df_override["coord_x"], df_override["coord_y"]]
        df_override["value_fail_off"] = game_value_fail_off[df_override["coord_x"], df_override["coord_y"]]
        df_override["value_success_def"] = game_value_success_def[df_override["coord_x"], df_override["coord_y"]]
        df_override["value_fail_def"] = game_value_fail_def[df_override["coord_x"], df_override["coord_y"]]
        df_override["value_success"] = df_override["value_success_off"] - df_override["value_success_def"]
        df_override["value_fail"] = df_override["value_fail_off"] - df_override["value_fail_def"]
        return df_override
    
