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

    unique_instance_str = str(uuid.uuid1())

    # req_body = json.loads(zlib.decompress(req.get_body()))

    # X_json = req_body['X']
    # y_json = req_body['y']

    # X = pd.read_json(X_json)
    # X = X.reindex(sorted(X.columns), axis=1)
    # y = pd.read_json(y_json, orient='index').sort_index()[0]

    X_file = req.files['X_file']
    y_file = req.files['y_file']

    tempFilePath = tempfile.gettempdir()
    staging_dir = tempFilePath + '/staging/' + unique_instance_str

    if not os.path.exists(staging_dir):
        logging.info('Creating '+staging_dir)
        os.makedirs(staging_dir)

    X_file.save(staging_dir+'/X.parquet')
    y_file.save(staging_dir+'/y.parquet')

    X = pd.read_parquet(staging_dir+'/X.parquet')
    X = X.reindex(sorted(X.columns), axis=1)
    # y = pd.read_parquet(staging_dir+'/y.parquet', orient='index').sort_index()[0]
    y = pd.read_parquet(staging_dir+'/y.parquet').iloc[:,0]

    os.remove(staging_dir+'/X.parquet')
    os.remove(staging_dir+'/y.parquet')
    os.rmdir(staging_dir)

    model = XGBRegressor()
    model.fit(X,y)

    # del req_body
    # del X_json
    # del y_json
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

    

    model.save_model(directory+'/model_'+unique_instance_str+'.json')
    logging.info('Model Saved Locally.')

    with open(directory+'/model_'+unique_instance_str+'.json') as f:
        model_json = json.load(f)
        logging.info('Model Read in Memory.')
    os.remove(directory+'/model_'+unique_instance_str+'.json')
    os.rmdir(directory)
    logging.info('Model Deleted from Local File System.')

    outp = {'model_json':model_json}
    logging.info('Output Ready to Send.')

    return func.HttpResponse(
        json.dumps(outp),
        mimetype='application/json',
    )
