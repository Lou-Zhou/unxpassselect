from typing import Callable
from socceraction.spadl.utils import add_names
import numpy as np
import pandas as pd
import torch
import itertools
from rich.progress import track
from torch.utils.data import DataLoader, Subset, random_split
from unxpass.components import pass_selection, pass_success, pass_value
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
        pass_value_success_component: pass_selection.SoccerMapComponent,
        pass_value_fail_component: pass_selection.SoccerMapComponent
    ):
        self.pass_value_success_component = pass_value_success_component
        self.pass_selection_component = pass_selection_component
        self.pass_success_component = pass_success_component
        self.pass_value_fail_component = pass_value_fail_component
        
    def rate_all_games(self, db, dataset, summarize = True, custom_pass = None):
        print("Generating Surfaces:")
        pass_selection_surface = self.pass_selection_component.predict_surface(dataset, db = db)
        pass_success_surface = self.pass_success_component.predict_surface(dataset,db = db)
        pass_value_surface_success = self.pass_value_success_component.predict_surface(dataset, db = db)
        pass_value_surface_fail = self.pass_value_fail_component.predict_surface(dataset, db = db)
        print("Finished.")
        alldf = []
        sels = self.pass_selection_component.predict(dataset)#ensuring that the actual pass made is in the dataset
        value_fail =self.pass_value_fail_component.predict(dataset)
        success = self.pass_success_component.predict(dataset)
        value_success = self.pass_value_success_component.predict(dataset)
        all_ratings = pd.concat([sels, success, value_success, value_fail], axis=1).rename(columns = {0:"selection_probability", 1: "success_probability", 2:"value_success", 3:"value_fail" })
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
                
                metrics = self.rate(game, action, pass_selection_surface, pass_success_surface, pass_value_surface_success, pass_value_surface_fail, db)
                metrics["Dist_From_True"] = (metrics["end_x"] - true_ends["end_x"])**2 + (metrics["end_y"] - true_ends["end_y"])**2
                closest_idx = np.where(metrics["Dist_From_True"] == min(metrics["Dist_From_True"]))[0]#getting closest pass to actual pass and then replacing that one with the original pass
                metrics = metrics.reset_index(drop = True)
                cols = ["end_x", "end_y", "selection_probability", "success_probability", "value_success", "value_fail"]
                metrics["True_Location"] = 0
                for col in cols:
                    metrics[col][closest_idx[0]] = game_ogs[col].loc[action]
                metrics["selection_probability"] = np.float64(metrics["selection_probability"])
                metrics["True_Location"][closest_idx[0]] = 1
                metrics["Expected_Utility"] = (metrics["success_probability"] * metrics["value_success"]) + ((1 - metrics["success_probability"]) * metrics["value_fail"])
                metrics["Utility*P(Selection)"] = metrics["selection_probability"] * metrics["Expected_Utility"]
                
                metrics["Sum(l')"] = sum(metrics["Utility*P(Selection)"]) - metrics["Utility*P(Selection)"]
                
                metrics["subSelCriteria"] = np.float64(metrics["selection_probability"]) * (metrics["Expected_Utility"] - metrics["Sum(l')"])**2
                metrics["Selection_Criteria"] = sum(metrics["subSelCriteria"])
                metrics = metrics.sort_values(by = ["subSelCriteria"], ascending = False)
                metrics["Evaluation_Criteria"] = metrics["True_Location"] * (metrics["Expected_Utility"]  - metrics["Sum(l')"])
                
                metrics = metrics.drop(columns = ["Dist_From_True"])
                if summarize:
                    metrics = metrics[metrics["True_Location"] == 1]
                    metrics = metrics[["original_event_id", "game_id", "action_id", "start_x","start_y", "end_x", "end_y", "result_id", "Selection_Criteria", "Evaluation_Criteria"]]
                if custom_pass is not None:
                    return metrics
                alldf.append(metrics)
            
        combined = pd.concat(alldf)
        return combined
    def rate(self, game, action, selection, success, value_success, value_fail, db):
        game_selection = selection[game][action]
        game_success = success[game][action]
        game_value_success = value_success[game][action]
        game_value_fail = value_fail[game][action]
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
        df_override["value_success"] = game_value_success[df_override["coord_x"], df_override["coord_y"]]
        df_override["value_fail"] = game_value_fail[df_override["coord_x"], df_override["coord_y"]]
        
        return df_override
    
