import xgboost as xgb
import os
import numpy as np
import pandas as pd
import awswrangler as wr
from fastapi import FastAPI, status, Request, Response
from typing import Union

best_threshold = 0.264033

app = FastAPI()

model_path = '/opt/ml/model'

# print([os.path.join(dirpath, f) for (dirpath, _, filenames) in os.walk(model_path) for f in filenames])

def load_model(path):
    booster = xgb.Booster()
    return booster.load_model(path)


def feature_calculation(users):
    weekly_p2p_df = wr.athena.read_sql_query(sql="SELECT * FROM weekly_p2p", database="feature_stores")
    return


def predict_output(body):
    booster = load_model('/opt/ml/model/cloned_user_detection.json')
    user_features = feature_calculation(body['users'])
    predicted_label = np.where(np.array([pred[1] for pred in booster.predict_proba(user_features)]) >= best_threshold, 1, 0).tolist()
    return zip(body['users'], predicted_label)


@app.post('/invocations')
async def invocations(request: Request):
    # model() is a hypothetical function that gets the inference output:
    model_resp = await predict_output(Request.json())
    print(model_resp)

    response = Response(
        content=model_resp,
        status_code=status.HTTP_200_OK,
        media_type="text/plain",
    )
    return response