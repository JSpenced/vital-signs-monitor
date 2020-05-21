# Vital Signs Monitor Deployment

Machine Learning model prediction and streaming api for monitoring health signals using AWS. The model is deployed using Serverless to generate AWS Lambda functions that are callable via the AWS API Gateway. The functions return a prediction if the users vitals are abnormal and dumps all the data into S3 buckets for later analysis or re-training of the models.

## Train Model

This will train a model for each user and save it into a S3 bucket.

### Set Up Project


```sh
cd vital-signs-monitor
conda create --name vital-signs python=3.8
conda activate vital-signs
pip install -r requirements.txt
```

### Train and Deploy user models to S3
First create the bucket using the aws cli and then train each user and store to the s3 bucket.

```sh
sh scripts/setup_bucket.sh
python src/model/train.py
```

## Deploy Prediction and S3 Storage API
Use serverless to deploy a Lambda function to store and return predictions of the current users' health status.

### Set Up Serverless
Install node.js and npm if it isn't installed before running the commands below.

```sh
cd src/deployment/prediction/
npm install -g serverless
sls plugin install -n serverless-python-requirements
```

### Deploy prediction API to AWS
Use serverless to deploy prediction and storage API using AWS API Gateway and AWS Lambda.

```sh
sls deploy -v
```

## Data Exploration
Notebooks of initial data exploration is in the notebooks folder.
