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
from dask.distributed import Client, wait
from xgboost.dask import DaskDMatrix

from sagemaker_algorithm_toolkit.exceptions import AlgorithmError, UserError
from sagemaker_xgboost_container.data_utils import CSV, PARQUET


def _read_data(local_path: str, content_type: str) -> (DataFrame, Series):
    if content_type == CSV:
        dataframe = dask_dataframe.read_csv(os.path.join(local_path, "*.csv"), header=None)
    elif content_type == PARQUET:
        dataframe = dask_dataframe.read_parquet(local_path)
    else:
        raise UserError(f"Unexpected content type '{content_type}'. Supported content types are CSV and PARQUET.")

    target_column = dataframe.columns[0]
    labels = dataframe[target_column]
    features = dataframe.drop(target_column, axis=1)

    return features, labels


def get_dataframe_dimensions(dataframe: DataFrame) -> (int, int):
    df_shape = dataframe.shape
    # Note that dataframe.shape[0].compute() is an expensive operation.
    rows = df_shape[0].compute()
    cols = df_shape[1]
    return rows, cols


def load_data_into_memory(client: Client, local_data_path: str, content_type: str) -> (DataFrame, Series):
    try:
        features, labels = _read_data(local_data_path, content_type)
        # Due to the lazy nature of Dask collections,
        # most data related errors will likely show up once data load is started here.
        features, labels = client.persist([features, labels])
        wait([features, labels])
    except Exception as e:
        raise UserError(f"Failed to load data. Exception: {e}")
    return features, labels


def create_dask_dmatrix(client: Client, features: DataFrame, labels: Series) -> DaskDMatrix:
    try:
        dmatrix = DaskDMatrix(client, features, labels)
    except Exception as e:
        raise AlgorithmError(f"Failed to create DaskDMatrix with given data. Exception: {e}")
    return dmatrix
