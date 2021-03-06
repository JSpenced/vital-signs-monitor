import json
import numpy as np
from sklearn.base import BaseEstimator
from src.utils import get_data_from_str, get_outlier_model, save_data_to_s3, get_scaler


def lambda_handler(event, context):
    """Lambda handler to ingest data, make prediction, and return prediction to client.
    """
    response = {"headers": {
        "Content-Type": "application/json"}}

    # Amazon API Gateway wraps input into a dictionary as the body
    # and local invocation uses raw input and does not wrap it
    if 'body' in event:
        data = get_data_from_str(event['body'])
    else:
        data = get_data_from_str(event)

    if not data:
        response['statusCode'] = 400
        response['body'] = json.dumps("Malformed data input.")
        return response

    save_success = save_data_to_s3(data)
    model = get_outlier_model(data['user_id'])
    scaler = get_scaler(data['user_id'])

    if model and scaler:
        response['statusCode'] = 200
        if data['hr'] == -1 and data['rr'] == -1:
            response['body'] = json.dumps(-10)
        else:
            scaled_data = scaler.transform(
                np.array([data['hr'], data['rr']]).reshape(1, -1))
            pred = predict(model, scaled_data)
            response['body'] = json.dumps(pred)
    else:
        response['statusCode'] = 404
        response['body'] = json.dumps(
            "Model or scaler not retrieved from S3 successfully.")

    return response


def predict(model: BaseEstimator, sample: list) -> int:
    """Make a prediction.

    Returns:
    A -1 indicating an outlier or 1 indicating a normal value.
    """

    # Need to reshape a single sample because input needs to be 2d
    result = model.predict(sample)
    return int(result[0])
