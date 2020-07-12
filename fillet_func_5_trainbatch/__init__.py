import logging
import os
import json
import pandas as pd
from xgboost import XGBRegressor
import azure.functions as func
import tempfile
import zlib, json, base64
import uuid
import gc

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    req_body = json.loads(zlib.decompress(req.get_body()))
    # req_body = json.loads(zlib.decompress(base64.b64decode(req.get_body())))

    # req_body = json.loads(req.get_body())

    X_json = req_body['X']
    y_json = req_body['y']

    X = pd.read_json(X_json)
    X = X.reindex(sorted(X.columns), axis=1)
    y = pd.read_json(y_json, orient='index').sort_index()[0]

    model = XGBRegressor()
    model.fit(X,y)

    del req_body
    del X_json
    del y_json
    del X
    del y

    gc.collect()
    
    tempFilePath = tempfile.gettempdir()

    logging.info('Model Trained Successfully.')
    logging.info(tempFilePath)

    directory = tempFilePath + '/data'

    if not os.path.exists(directory):
        logging.info('Creating '+directory)
        os.makedirs(directory)

    unique_instance_str = str(uuid.uuid1())

    model.save_model(directory+'/model_'+unique_instance_str+'.json')
    logging.info('Model Saved Locally.')

    with open(directory+'/model_'+unique_instance_str+'.json') as f:
        model_json = json.load(f)
        logging.info('Model Read in Memory.')
    os.remove(directory+'/model_'+unique_instance_str+'.json')
    logging.info('Model Deleted from Local File System.')

    outp = {'model_json':model_json}
    logging.info('Output Ready to Send.')

    return func.HttpResponse(
        json.dumps(outp),
        mimetype='application/json',
    )
