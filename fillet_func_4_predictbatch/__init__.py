import logging
import os
import json
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import azure.functions as func
import tempfile
import uuid
import pickle as p


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    data_dict = p.loads(req.get_body())


    prices_json = data_dict['prices']
    models_list = data_dict['models']

    prices = pd.read_json(prices_json)
    prices = prices.reindex(sorted(prices.columns), axis=1)
    

    qty_est_list = []

    for model in models_list:
        prices = prices[model.get_booster().feature_names]
        pred = model.predict(data=prices)



        pred_value = str(pred[0])
        qty_est_list.append(pred_value)


    outp = {
        'qty_estimates': qty_est_list,
    }

    return func.HttpResponse(
        json.dumps(outp),
        mimetype='application/json',
    )
