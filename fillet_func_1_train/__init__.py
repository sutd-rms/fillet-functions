import logging
import os
import json
import pandas as pd
from xgboost import XGBRegressor
import azure.functions as func
import tempfile
import uuid
import gc
import pickle

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    unique_instance_str = str(uuid.uuid1())

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
    y = pd.read_parquet(staging_dir+'/y.parquet').iloc[:,0]

    os.remove(staging_dir+'/X.parquet')
    os.remove(staging_dir+'/y.parquet')
    os.rmdir(staging_dir)

    model = XGBRegressor()
    model.fit(X,y)

    del X
    del y

    gc.collect()

    model_pickle = pickle.dumps(model)

    logging.info('Model Saved Locally.')

    return func.HttpResponse(
        body=model_pickle,
    )
