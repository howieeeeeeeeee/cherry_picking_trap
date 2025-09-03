import os


from import_setup import *
from src.simulation.cherry_picking_model import CherryPickingModel
import numpy as np
from src.data_management.mongo_handler import insert_simulation_result
from config.db import MONGOCLIENT_LOCAL

db = MONGOCLIENT_LOCAL["cherry_picking_trap"]

Ks = [i for i in range(1, 50, 1)]
for i in range(50, 1050, 50):
    Ks += [i]

theta = 2.0
for lamb in np.linspace(0, 1, 11):
    lamb = round(lamb, 1)
    for scale in [0.5, 1, 1.5, 2, 3]:
        for K in Ks:
            dic = {"lamb": lamb, "K": K, "scale": scale}
            ## check whether the result already exists
            if db["simulation_results"].find_one(
                {"lamb": lamb, "K": K, "scale": scale}
            ):
                continue
            try:
                model = CherryPickingModel(
                    K=K,
                    lamb=lamb,
                    theta=theta,
                    distribution_name="exponential",
                    dist_params={"scale": scale},  ## scale = \mu
                )
                result = model.solve_and_analyze(num_simulations=1_000_000)
                dic.update({"success": True})
                dic.update(result)
            except Exception as e:
                print(e)
                dic.update({"success": False})
                dic.update({"error": str(e)})
                continue

            result.update(
                {
                    "lamb": lamb,
                    "K": K,
                    "scale": scale,
                    "dist": "exponential",
                    "theta": theta,
                }
            )
            insert_simulation_result(db, result)
