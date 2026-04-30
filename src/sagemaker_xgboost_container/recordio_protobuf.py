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
"""RecordIO-Protobuf reader without MLIO dependency.

Uses the SageMaker aialgs protobuf schema (record_pb2) from sagemaker_containers
to properly decode RecordIO-framed protobuf records.
"""
import struct

import numpy as np
from scipy.sparse import csr_matrix, vstack as scipy_vstack

from sagemaker_containers.record_pb2 import Record

# RecordIO magic number (4 bytes little-endian)
_RECORDIO_MAGIC = 0xCED7230A


def _read_recordio_records(buf):
    """Yield individual protobuf record bytes from a RecordIO byte buffer."""
    offset = 0
    while offset < len(buf):
        if offset + 8 > len(buf):
            break
        magic, length = struct.unpack_from("<II", buf, offset)
        if magic != _RECORDIO_MAGIC:
            raise ValueError(f"Invalid RecordIO magic at offset {offset}")
        offset += 8
        padded_length = ((length + 3) // 4) * 4
        if offset + padded_length > len(buf):
            raise ValueError(f"Truncated record at offset {offset}")
        yield buf[offset:offset + length]
        offset += padded_length


def _extract_tensor(value):
    """Extract numpy array and optional keys/shape from a Value protobuf message.

    Returns (values_array, keys_array_or_None, shape_list_or_None, is_sparse)
    """
    if value.HasField("float32_tensor"):
        t = value.float32_tensor
        values = np.array(t.values, dtype=np.float32)
        keys = np.array(t.keys, dtype=np.uint64) if t.keys else None
        shape = list(t.shape) if t.shape else None
        return values, keys, shape, keys is not None
    elif value.HasField("float64_tensor"):
        t = value.float64_tensor
        values = np.array(t.values, dtype=np.float64)
        keys = np.array(t.keys, dtype=np.uint64) if t.keys else None
        shape = list(t.shape) if t.shape else None
        return values, keys, shape, keys is not None
    elif value.HasField("int32_tensor"):
        t = value.int32_tensor
        values = np.array(t.values, dtype=np.int32)
        keys = np.array(t.keys, dtype=np.uint64) if t.keys else None
        shape = list(t.shape) if t.shape else None
        return values, keys, shape, keys is not None
    return None, None, None, False


def read_recordio_protobuf(buf):
    """Read RecordIO-Protobuf data and return features and labels.

    Args:
        buf (bytes): Raw RecordIO-Protobuf byte buffer.

    Returns:
        tuple: (features, labels) where features is numpy array or scipy csr_matrix,
               and labels is numpy array or None.
    """
    all_features = []
    all_labels = []
    is_sparse = False

    for record_bytes in _read_recordio_records(buf):
        record = Record()
        record.ParseFromString(record_bytes)

        # Extract features (key is typically "values")
        if "values" in record.features:
            values, keys, shape, sparse = _extract_tensor(record.features["values"])
            if values is None and keys is None:
                # Empty record — create zero-row based on shape
                if shape:
                    ncols = int(shape[0])
                    is_sparse = True
                    row = csr_matrix((1, ncols), dtype=np.float32)
                    all_features.append(row)
                continue

            if values is None:
                values = np.array([], dtype=np.float32)

            if sparse:
                is_sparse = True
                if shape:
                    ncols = int(shape[0])
                elif keys is not None and len(keys) > 0:
                    ncols = int(keys.max()) + 1
                else:
                    ncols = 1
                if keys is None:
                    keys = np.array([], dtype=np.int64)
                row = csr_matrix(
                    (values, keys.astype(np.int64), [0, len(keys)]),
                    shape=(1, ncols),
                )
                all_features.append(row)
            else:
                all_features.append(values.reshape(1, -1))
        else:
            continue

        # Extract labels (key is typically "values" in label map)
        if "values" in record.label:
            label_values, _, _, _ = _extract_tensor(record.label["values"])
            if label_values is not None:
                all_labels.append(label_values)

    if not all_features:
        raise ValueError("No records found in RecordIO-Protobuf data")

    if is_sparse:
        result_features = scipy_vstack(all_features).tocsr()
    else:
        result_features = np.vstack(all_features)

    result_labels = np.concatenate(all_labels, axis=None) if all_labels else None

    return result_features, result_labels
