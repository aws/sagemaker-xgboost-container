# XGen SageMaker XGBoost Container
A SageMaker XGBoost Container that is used for Inference Endpoint

### AWS ECR login
```
aws ecr get-login-password --profile dev --region us-east-1 | docker login --username AWS --password-stdin {ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com
```

# Build
## Steps
### Base Image
```
docker build -t xgboost-container-base:0.90-2-cpu-py3 -f docker/0.90-2/base/Dockerfile.cpu .
```

### Setup Dependencies
```
python setup.py bdist_wheel --universal
```

### Final Image
```
docker build -q -t multi-model-xgboost -f docker/0.90-2/final/Dockerfile.cpu .
```

### Tag the Image
```
docker tag multi-model-xgboost "{ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/multi-model-xgboost:latest"
```

### Push the image into AWS ECR
```
docker push {ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/multi-model-xgboost:latest
```

# Update Endpoint
### Caveat
Currently, SageMaker Endpoint does not automatically update upon the new push of ECR image. As a short term solution, we create a new configuration file that points to the latest pushed ECR image, and make the endpoint to point to it to trigger the update.

## Steps
1. Go to AWS SageMaker Console -> Endpoint configurations
2. 'Clone' the latest Endpoint configuration with a prefix "multi-model-xgboost-config-*"
3. Set the 'Endpoint configuration name' to "multi-model-xgboost-config-copy-{MM}-{DD}" (Most of it will already be generated)
4. From 'Production variants', check:
   - if the model is pointing to the correct ECR image
   - other configurations are correct
5. 'Create endpoint configuration'
6. Go to AWS SageMaker Console -> Endpoints
7. Click on the `multi-model-xgboost` and 'Update endpoint'
8. 'Change Endpoint configuration' -> 'Use an existing endpoint configuration' -> Choose the configuration created from Step 5 -> 'Select endpoint configuration' -> 'Update endpoint'
9. Wait until the Status = `InService`

# TESTING
## Configuration Change
### Steps
1. Go to AWS CloudWatch Console -> Insights -> 'Select log group(s)' -> `/aws/sagemaker/Endpoints/multi-model-xgboost`
2. Query following for last 5 mins
```
fields @timestamp, @message
| filter 
@message like "CPU" or 
@message like "GPU" or 
@message like "workers" or
@message like {WHATEVER METRICS OF INTEREST}
| sort @timestamp desc
| limit 20
```
3. Observe the values are shown as expected.
Ex.
```
1
2021-02-05T10:06:17.014-08:00
Number of GPUs: 0
2
2021-02-05T10:06:17.014-08:00
Number of CPUs: 2
3
2021-02-05T10:06:17.014-08:00
Default workers per model: 16
```
## Test the Inference Locally
### Steps
Test script: `xgen/x2mind/test/test_inference.py`
1. Set the `customer_id`, `model_id`, `user_id` with valid Customer ID, Model ID, User ID respectively.
2. Run the script by `python3 test_inference.py`
3. Go to AWS CloudWatch Console -> Insights -> 'Select log group(s)' -> `/aws/sagemaker/Endpoints/multi-model-xgboost`
2. Query following:
```
fields @timestamp, @message
| filter 
@message like "Error" or 
@message like {WHATEVER METRICS OF INTEREST}
| sort @timestamp desc
| limit 20
```
3. Observe the values are shown as expected.
