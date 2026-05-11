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

import glob
import os

import dask.array as da
import numpy as np
import pandas as pd
from dask.distributed import Client
from xgboost import dask as dxgb

from sagemaker_algorithm_toolkit.exceptions import AlgorithmError, UserError
from sagemaker_xgboost_container.data_utils import CSV, PARQUET


def read_data(local_path: str, content_type: str):
    """Read training data into pandas DataFrame.

    Uses pandas directly instead of dask.dataframe to avoid expression
    optimizer bug in dask >= 2026.1 (ValueError on column comparison).
    """
    if content_type == CSV:
        files = sorted(glob.glob(os.path.join(local_path, "*.csv")))
        pdf = pd.concat([pd.read_csv(f, header=None) for f in files], ignore_index=True)
    elif content_type == PARQUET:
        files = sorted(glob.glob(os.path.join(local_path, "*.parquet")))
        pdf = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    else:
        raise UserError(
            f"Unexpected content type '{content_type}'. Supported content types are CSV and PARQUET."
        )

    target_column = pdf.columns[0]
    labels = pdf[target_column]
    features = pdf[pdf.columns[1:]]

    return features, labels


def get_dataframe_dimensions(dataframe) -> (int, int):
    if hasattr(dataframe, 'compute'):
        rows = dataframe.shape[0].compute()
        cols = dataframe.shape[1]
    else:
        rows, cols = dataframe.shape
    return rows, cols


def create_dask_dmatrix(client: Client, features, labels) -> dxgb.DaskDMatrix:
    """Create DaskDMatrix from pandas DataFrame/Series or numpy arrays.

    Converts to numpy and wraps in dask.array with chunks distributed
    across workers to bypass the dask.dataframe expression optimizer bug
    in dask >= 2026.1, while preserving multi-worker parallelism.
    """
    try:
        features_np = features.values if hasattr(features, 'values') else np.asarray(features)
        labels_np = labels.values if hasattr(labels, 'values') else np.asarray(labels)

        # Chunk rows across available workers for distributed training
        n_workers = max(1, len(client.scheduler_info()['workers']))
        n_rows = features_np.shape[0]
        row_chunks = max(1, n_rows // n_workers)

        features_da = da.from_array(features_np, chunks=(row_chunks, features_np.shape[1]))
        labels_da = da.from_array(labels_np, chunks=(row_chunks,))
        dmatrix = dxgb.DaskDMatrix(client, features_da, labels_da)
    except Exception as e:
        raise AlgorithmError(
            f"Failed to create DaskDMatrix with given data. Exception: {e}"
        )
    return dmatrix
