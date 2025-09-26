# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import, print_function

import argparse
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument("--objective", type=str, default="reg:squarederror")
    parser.add_argument("--colsample-bytree", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--reg-alpha", type=int, default=10)
    parser.add_argument("--n-estimators", type=int, default=10)
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])

    args = parser.parse_args()

    # Load the California housing data into pandas data frame (replacement for deprecated Boston dataset)
    california = fetch_california_housing()
    data = pd.DataFrame(california.data)
    data.columns = california.feature_names
    data["PRICE"] = california.target

    # Convert Pandas dataframe to XGBoost DMatrix for better performance (used later).
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    data_dmatrix = xgb.DMatrix(data=X, label=y)

    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

    # Create regressor object by using SKLearn API
    xg_reg = xgb.XGBRegressor(
        objective=args.objective,
        colsample_bytree=args.colsample_bytree,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        reg_alpha=args.reg_alpha,
        n_estimators=args.n_estimators,
    )

    # Train and save the model
    xg_reg.fit(X_train, y_train)
    model_path = os.path.join(args.model_dir, "xgb-california.model")
    xg_reg.get_booster().save_model(model_path)

    # Make predictions and calculate RMSE
    preds = xg_reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print("RMSE: %f" % (rmse))

    # We can look at the feature importance and store the graph as an image.
    if not os.path.exists(args.output_data_dir):
        os.makedirs(args.output_data_dir)

    ax = xgb.plot_importance(xg_reg)
    fig = ax.figure
    fig.set_size_inches(5, 5)
    fig.savefig(os.path.join(args.output_data_dir, "feature-importance-plot.png"))

    # Finally, lets do a bit of cross-validation by using native XGB functionality (keeping some parameters constant, so
    # that we don't have a huge input list for this simple example.
    params = {
        "objective": args.objective,
        "colsample_bytree": args.colsample_bytree,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "alpha": args.reg_alpha,
    }
    cv_results = xgb.cv(
        dtrain=data_dmatrix,
        params=params,
        nfold=5,
        num_boost_round=50,
        early_stopping_rounds=10,
        metrics="rmse",
        as_pandas=True,
        seed=100,
    )

    cv_results.to_csv(os.path.join(args.output_data_dir, "cv_results.csv"))