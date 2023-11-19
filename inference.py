import xgboost as xgb
import os

model_xgb_2 = xgb.Booster()
model_xgb_2.load_model("model.json")

model_path = '/opt/ml/model'

print([os.path.join(dirpath, f) for (dirpath, _, filenames) in os.walk(model_path) for f in filenames])

def load_model(path):
    return


def predict():
    return


