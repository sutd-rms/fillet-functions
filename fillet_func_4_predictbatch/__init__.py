import logging
import os
import json
import pandas as pd
import numpy as np
from xgboost import XGBRegressor, Booster, DMatrix
import azure.functions as func
import tempfile
import uuid
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import zlib


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # req_body = json.loads(zlib.decompress(base64.b64decode(req.get_body())))

    # X_json = req_body['X']
    # y_json = req_body['y']
    # Week_json = req_body['Week']

    # req_body = json.loads(req.get_body())
    req_body = json.loads(zlib.decompress(req.get_body()))

    prices_json = req_body['prices']
    models_list = req_body['models']

    prices = pd.read_json(prices_json)
    prices = prices.reindex(sorted(prices.columns), axis=1)
    # prices_dmatrix = DMatrix(prices)

    qty_est_list = []

    for model_json in models_list:

        # Save Model to Temp
        unique_instance_str = str(uuid.uuid1())
        tempFilePath = tempfile.gettempdir()
        directory = tempFilePath + '/model_temp'
        if not os.path.exists(directory):
            logging.info('Creating '+directory)
            os.makedirs(directory)

        with open(directory+'/model_'+unique_instance_str+'.json', 'w') as f:
            json.dump(model_json, f)

        # Load Model
        xgb = XGBRegressor()
        xgb.load_model(directory+'/model_'+unique_instance_str+'.json')
        # bst = Booster()
        # bst.load_model(directory+'/model_'+unique_instance_str+'.json')
        os.remove(directory+'/model_'+unique_instance_str+'.json')

        # Make Prediction
        # pred = bst.predict(data=prices_dmatrix)
        pred = xgb.predict(data=prices)
        pred_value = str(pred[0])
        qty_est_list.append(pred_value)


    outp = {
        'qty_estimates': qty_est_list,
    }

    return func.HttpResponse(
        json.dumps(outp),
        mimetype='application/json',
    )
