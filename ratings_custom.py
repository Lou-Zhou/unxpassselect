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
        pass_success_component: pass_success.XGBoostComponent,
        pass_value_success_component: pass_selection.SoccerMapComponent,
        pass_value_fail_component: pass_selection.SoccerMapComponent
    ):
        self.pass_value_success_component = pass_value_success_component
        self.pass_selection_component = pass_selection_component
        self.pass_success_component = pass_success_component
        self.pass_value_fail_component = pass_value_fail_component
    def get_all(self, db, dataset: Callable, game_id, action_id):
        pass_selection_surface = self.pass_selection_component.predict_surface(dataset)
        surface_action = pass_selection_surface[game_id][action_id]
        x_lim, y_lim = surface_action.shape
        coords = list(itertools.product(range(0,x_lim), range(0,y_lim)))
        base = 0
        limit = self.pass_success_component.initialize_dataset(dataset).features.shape[0]
        og_limit = limit
        all_ratings = []
        while base < len(coords):
            print(base, limit)
            #ratings, surface = self.rate(db, dataset, game_id, action_id, base, limit, coords, x_lim, y_lim, surface_action)
            ratings = self.rate(db, dataset, game_id, action_id, base, limit, coords, x_lim, y_lim, surface_action)
            base = base + og_limit
            limit = limit + og_limit
            all_ratings.append(ratings)
        all_locations= pd.concat(all_ratings)
        return all_locations
    def rate(self, db, dataset: Callable, game_id, action_id, base, limit, coords, x_lim ,y_lim, surface):
        data = dataset(
            xfns={
                "startlocation": ["start_x_a0", "start_y_a0"],
                "endlocation": ["end_x_a0", "end_y_a0"],
            },
            yfns=["success"],
        )
        
        data_pass_success = self.pass_success_component.initialize_dataset(dataset)
        test_db = db.actions(game_id = game_id)
        if limit > len(coords):
            num_options = len(coords) - base
        else:
            num_options  = data_pass_success.features.shape[0]
        df_override = pd.DataFrame([test_db.loc[(game_id, action_id)]] * num_options)
        df_override["coord_x"] = [i[0] for i in coords][base:limit]
        df_override["coord_y"] = [i[1] for i in coords][base:limit]
        df_override["end_x"] = convert_x(df_override["coord_x"], x_lim)
        df_override["end_y"] = convert_y(df_override["coord_y"], y_lim)
        df_override.index = data_pass_success.features.head(num_options).index
        df_override["true_game_id"] = game_id 
        df_override["true_action_id"] = action_id
        feat_typical_pass_success = data_pass_success.apply_overrides(
            db,
            df_override
            )
        
        feat_typical_pass_success.limit_dataset(num_options)
        df_override["pass_success"] = self.pass_success_component.predict(
             feat_typical_pass_success
        )
        data_pass_value = self.pass_value_success_component.initialize_dataset(dataset)
        feat_typical_pass_value_success = data_pass_value.apply_overrides(
            db,
            df_override,
            )
        feat_typical_pass_value_success.limit_dataset(num_options)
        #feat_typical_pass_value_success.labels["success"] = False
        #return feat_typical_pass_value_success.features
        df_override["typical_value_success"] = self.pass_value_success_component.new_predict(
            feat_typical_pass_value_success
            )
        #return feat_typical_pass_value_success.labels, df_override["typical_value_success"]
        feat_typical_pass_value_fail = data_pass_value.apply_overrides(
        db,
        df_override,
        )
        #feat_typical_pass_value_fail.labels["success"] = True
        feat_typical_pass_value_fail.limit_dataset(num_options)
        df_override["typical_value_fail"] = self.pass_value_fail_component.new_predict(
        feat_typical_pass_value_fail
            )
        df_override["selection_probability"] = surface[df_override["coord_x"], df_override["coord_y"]]
        return df_override
    
