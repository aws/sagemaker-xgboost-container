import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import tarfile
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, roc_curve
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import boto3
from botocore.exceptions import ClientError

print('hello world!')

data_prepared_df = pd.read_parquet('s3://mlops-feature-stores/data-prepared')

data_prepared_df = data_prepared_df[(data_prepared_df.created_at >= '2022-06-01')
                                     & (data_prepared_df.created_at <= '2022-10-31')
                                       & (data_prepared_df.status == 'active')]

cats = data_prepared_df.select_dtypes(exclude=np.number).columns.tolist()

for col in cats:
    if col.endswith('trading_amount') or col.endswith('per_transaction'):
        data_prepared_df[col] = data_prepared_df[col].astype('float32')
    else:
        data_prepared_df[col].fillna('TBD', inplace=True)
        data_prepared_df[col] = data_prepared_df[col].astype('category')

X, y = data_prepared_df.drop(['user_id',
                              'weekly_report',
                              'monthly_report',
                              'is_cloned',
                              'created_at',
                              'status',
                              'username',
                              'label'], axis=1), data_prepared_df[['label']]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

#Training data
xgb_classifier = XGBClassifier(n_estimators=100,
                               objective='binary:logistic',
                               tree_method='hist',
                               eta=0.1,
                               max_depth=6,
                               verbosity=3,
                               n_jobs=-1,
                               subsample=0.8,
                               enable_categorical=True)

xgb_classifier.fit(X_train, y_train)

model_path_basename = '/src/ml_model'
if not os.path.exists(model_path_basename):
    os.makedirs(model_path_basename)

model_path = os.path.join(model_path_basename, 'cloned_user_detection.json')

output_filename = 'cloned_user_detection.tar.gz'

arcname = os.path.basename(model_path)

xgb_classifier.save_model(model_path)

with tarfile.open(output_filename, "w:gz") as tar:
    tar.add(model_path, arcname=arcname)

s3_path = 's3://mlops-feature-stores/models/cloned-user-detection'

s3_client = boto3.client('s3', region_name='us-east-1')

try:
    response = s3_client.upload_file(output_filename, 'mlops-feature-stores', 'models/cloned-user-detection/cloned_user_detection.tar.gz')
except ClientError as e:
    print(e)



