from typing import Callable

import numpy as np
import pandas as pd

from unxpass.components import pass_selection, pass_success, pass_value
from unxpass.datasets import PassesDataset

##My version of unxpass/components/ratings.py, converted to predict all x and y locations
def typical_pass(pass_selection_surface, x_t, y_t):
    """Get typical pass"""
    #edited to convert selection surface coords to real x and y
    p_odds = pass_selection_surface[y_t, x_t]
    y_dim, x_dim = pass_selection_surface.shape
    y_t = y_t / y_dim * 68 + 68 / y_dim / 2
    x_t = x_t / x_dim * 105 + 105 / x_dim / 2
    
    return x_t, y_t, p_odds


def _get_cell_indexes(x, y, x_bins=104, y_bins=68):
    x_bin = np.clip(x / 105 * x_bins, 0, x_bins - 1).astype(np.uint8)
    y_bin = np.clip(y / 68 * y_bins, 0, y_bins - 1).astype(np.uint8)
    return x_bin, y_bin


class CreativeDecisionRating:
    def __init__(
        self,
        pass_selection_component: pass_selection.SoccerMapComponent,
        pass_success_component: pass_success.XGBoostComponent,
        pass_value_component: pass_value.VaepModel,
    ):
        self.pass_value_component = pass_value_component
        self.pass_selection_component = pass_selection_component
        self.pass_success_component = pass_success_component
    
    def rate(self, db, dataset: Callable, x_t, y_t):
        # get the actual start and end location of each pass
        data = dataset(
            xfns={
                "startlocation": ["start_x_a0", "start_y_a0"],
                "endlocation": ["end_x_a0", "end_y_a0"],
            },
            yfns=["success"],
        )
        df_ratings = pd.concat([data.features, data.labels], axis=1).rename(
            columns={
                "start_x_a0": "start_x",
                "start_y_a0": "start_y",
                "end_x_a0": "true_end_x",
                "end_y_a0": "true_end_y",
            }
        )

        # get pass selection probabilities
        pass_selection_surfaces = self.pass_selection_component.predict_surface(dataset)

        # sets typical_end_x and typical_end_y to specificed x and y coordinates
        for game_id in pass_selection_surfaces:
            for action_id in pass_selection_surfaces[game_id]:
                surface = pass_selection_surfaces[game_id][action_id]
                df_ratings.loc[
                    (game_id, action_id), ["typical_end_x", "typical_end_y", "selection_odds"]
                ] = typical_pass(surface, x_t, y_t)
        # get pass success probabilities
        data_pass_success = self.pass_success_component.initialize_dataset(dataset)
        feat_typical_pass_succes = data_pass_success.apply_overrides(
            db,
            df_ratings[["typical_end_x", "typical_end_y"]].rename(
                columns={"typical_end_x": "end_x", "typical_end_y": "end_y"}
            ),
        )
        df_ratings["typical_p_success"] = self.pass_success_component.predict(
            feat_typical_pass_succes
        )

        # get pass value
        data_pass_value = self.pass_value_component.offensive_model.initialize_dataset(dataset)

        feat_typical_pass_value_succes = data_pass_value.apply_overrides(
            db,
            df_ratings[["typical_end_x", "typical_end_y"]]
            .rename(columns={"typical_end_x": "end_x", "typical_end_y": "end_y"})
            .assign(result_id=1, result_name="success"),
        )
        df_ratings["typical_value_success"] = self.pass_value_component.predict(
            feat_typical_pass_value_succes
        )

        feat_typical_pass_value_fail = data_pass_value.apply_overrides(
            db,
            df_ratings[["typical_end_x", "typical_end_y"]]
            .rename(columns={"typical_end_x": "end_x", "typical_end_y": "end_y"})
            .assign(result_id=0, result_name="fail"),
        )
        df_ratings["typical_value_fail"] = self.pass_value_component.predict(
            feat_typical_pass_value_fail
        )

        return df_ratings
