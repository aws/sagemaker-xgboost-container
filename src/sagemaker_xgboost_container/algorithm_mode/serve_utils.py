import numpy as np
import os
import pickle as pkl
from scipy.sparse import csr_matrix
import xgboost as xgb

from sagemaker_xgboost_container import encoder
from sagemaker_xgboost_container.data_utils import CSV, LIBSVM, RECORDIO_PROTOBUF, get_content_type

PKL_FORMAT = 'pkl_format'
XGB_FORMAT = 'xgb_format'


# FIXME: https://github.com/aws/sagemaker-xgboost-container/issues/12
# This was failing a previous container test; need to make decision as to change test or behavior.
def _get_sparse_matrix_from_libsvm(payload):
    pylist = map(lambda x: x.split(' '), payload.split('\n'))
    colon = ':'
    row = []
    col = []
    data = []
    for row_idx, line in enumerate(pylist):
        for item in line:
            if colon in item:
                col_idx = item.split(colon)[0]
                val = item.split(colon)[1]
                row.append(row_idx)
                col.append(col_idx)
                data.append(val)

    row = np.array(row)
    col = np.array(col).astype(np.int)
    data = np.array(data).astype(np.float)
    if not (len(row) == len(col) and len(col) == len(data)):
        raise RuntimeError("Dimension checking failed when transforming sparse matrix.")

    return csr_matrix((data, (row, col)))


def parse_content_data(input_data, input_content_type):
    dtest = None
    content_type = get_content_type(input_content_type)
    payload = input_data
    if content_type == CSV:
        try:
            decoded_payload = payload.strip().decode("utf-8")
            dtest = encoder.csv_to_dmatrix(decoded_payload, dtype=np.float)
        except Exception as e:
            raise RuntimeError("Loading csv data failed with Exception, "
                               "please ensure data is in csv format:\n {}\n {}".format(type(e), e))
    elif content_type == LIBSVM:
        try:
            decoded_payload = payload.strip().decode("utf-8")
            dtest = xgb.DMatrix(_get_sparse_matrix_from_libsvm(decoded_payload))
        except Exception as e:
            raise RuntimeError("Loading libsvm data failed with Exception, "
                               "please ensure data is in libsvm format:\n {}\n {}".format(type(e), e))
    elif content_type == RECORDIO_PROTOBUF:
        try:
            dtest = encoder.recordio_protobuf_to_dmatrix(payload)
        except Exception as e:
            raise RuntimeError("Loading recordio-protobuf data failed with "
                               "Exception, please ensure data is in "
                               "recordio-protobuf format: {} {}".format(type(e), e))
    else:
        raise RuntimeError("Content-type {} is not supported.".format(input_content_type))

    return dtest, content_type


def get_loaded_booster(model_dir):
    model_files = (data_file for data_file in os.listdir(model_dir)
                   if os.path.isfile(os.path.join(model_dir, data_file)))
    model_file = next(model_files)
    try:
        booster = pkl.load(open(os.path.join(model_dir, model_file), 'rb'))
        format = PKL_FORMAT
    except Exception as exp_pkl:
        try:
            booster = xgb.Booster()
            booster.load_model(os.path.join(model_dir, model_file))
            format = XGB_FORMAT
        except Exception as exp_xgb:
            raise RuntimeError("Model at {} cannot be loaded:\n{}\n{}".format(model_dir, str(exp_pkl), str(exp_xgb)))
    booster.set_param('nthread', 1)
    return booster, format


def predict(booster, model_format, dtest, input_content_type):
    if model_format == PKL_FORMAT:
        x = len(booster.feature_names)
        y = len(dtest.feature_names)

        try:
            content_type = get_content_type(input_content_type)
        except Exception:
            raise ValueError('Content type {} is not supported'.format(input_content_type))

        if content_type == LIBSVM:
            if y > x + 1:
                raise ValueError('Feature size of libsvm inference data {} is larger than '
                                 'feature size of trained model {}.'.format(y, x))
        elif content_type in [CSV, RECORDIO_PROTOBUF]:
            if not ((x == y) or (x == y + 1)):
                raise ValueError('Feature size of {} inference data {} is not consistent '
                                 'with feature size of trained model {}.'.
                                 format(content_type, y, x))
        else:
            raise ValueError('Content type {} is not supported'.format(content_type))
    return booster.predict(dtest,
                           ntree_limit=getattr(booster, "best_ntree_limit", 0),
                           validate_features=False)
