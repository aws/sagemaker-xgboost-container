# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License'). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the 'license' file accompanying this file. This file is
# distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import os

import dask.dataframe as dask_dataframe
from dask.dataframe import DataFrame, Series
from dask.distributed import Client
from xgboost.dask import DaskDMatrix

from sagemaker_algorithm_toolkit.exceptions import AlgorithmError, UserError
from sagemaker_xgboost_container.data_utils import CSV, PARQUET


def read_data(local_path: str, content_type: str) -> (DataFrame, Series):
    if content_type == CSV:
        dataframe = dask_dataframe.read_csv(os.path.join(local_path, "*.csv"), header=None)
    elif content_type == PARQUET:
        dataframe = dask_dataframe.read_parquet(local_path)
    else:
        raise UserError(f"Unexpected content type '{content_type}'. Supported content types are CSV and PARQUET.")

    target_column = dataframe.columns[0]
    labels = dataframe[target_column]
    features = dataframe[dataframe.columns[1:]]

    return features, labels


def get_dataframe_dimensions(dataframe: DataFrame) -> (int, int):
    df_shape = dataframe.shape
    # Note that dataframe.shape[0].compute() is an expensive operation.
    rows = df_shape[0].compute()
    cols = df_shape[1]
    return rows, cols


def create_dask_dmatrix(client: Client, features: DataFrame, labels: Series) -> DaskDMatrix:
    try:
        dmatrix = DaskDMatrix(client, features, labels)
    except Exception as e:
        raise AlgorithmError(f"Failed to create DaskDMatrix with given data. Exception: {e}")
    return dmatrix
