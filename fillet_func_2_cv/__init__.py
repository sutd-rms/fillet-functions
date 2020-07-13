import logging
import os
import json
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import azure.functions as func
import tempfile
import zlib, json, base64
import gc
import uuid

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    unique_instance_str = str(uuid.uuid1())

    X_file = req.files['X_file']
    y_file = req.files['y_file']
    Wk_file = req.files['Wk_file']

    tempFilePath = tempfile.gettempdir()
    staging_dir = tempFilePath + '/staging/' + unique_instance_str

    if not os.path.exists(staging_dir):
        logging.info('Creating '+staging_dir)
        os.makedirs(staging_dir)

    X_file.save(staging_dir+'/X.parquet')
    y_file.save(staging_dir+'/y.parquet')
    Wk_file.save(staging_dir+'/Wk.parquet')

    X = pd.read_parquet(staging_dir+'/X.parquet')
    X = X.reindex(sorted(X.columns), axis=1)
    y = pd.read_parquet(staging_dir+'/y.parquet').iloc[:,0]
    Week = pd.read_parquet(staging_dir+'/Wk.parquet').iloc[:,0]

    os.remove(staging_dir+'/X.parquet')
    os.remove(staging_dir+'/y.parquet')
    os.remove(staging_dir+'/Wk.parquet')
    os.rmdir(staging_dir)



    logo = LeaveOneGroupOut()
    n_splits = logo.get_n_splits(groups=Week)

    r2_total = 0
    mae_total = 0
    rmse_total = 0
    logging.info('Beginning CV.')
    c=0

    target_splits = 4
    n_actual_splits = 0
    nth_split = 0

    for train_index, test_index in logo.split(X, y, Week):
        
        cv_prob = max(0,(target_splits - n_actual_splits)/(n_splits-nth_split))
        nth_split += 1

        if np.random.rand()>cv_prob:
            continue

        logging.info('Split {}.'.format(c))

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        y_train = np.asarray(y_train).ravel()
        y_test = np.asarray(y_test).ravel()

        test_model = XGBRegressor()
        test_model.fit(X_train, y_train)
        y_pred = test_model.predict(X_test)

        r2_total += r2_score(y_true=y_test, y_pred=y_pred)
        mae_total += mean_absolute_error(y_true=y_test, y_pred=y_pred)
        rmse_total += np.sqrt(
            mean_squared_error(y_true=y_test, y_pred=y_pred))
        n_actual_splits += 1
        c+=1

    avg_sales = y.mean()
    r2 = r2_total / n_actual_splits
    mae = mae_total / n_actual_splits
    mpe = mae / avg_sales
    rmse = rmse_total / n_actual_splits

    del X
    del y
    del Week

    gc.collect()

    outp = {
        'avg_sales': float(avg_sales),
        'r2_score': float(r2),
        'mae_score': float(mae),
        'mpe_score': float(mpe),
        'rmse_score': float(rmse)
    }

    return func.HttpResponse(
        json.dumps(outp),
        mimetype='application/json',
    )
