import json
import numpy as np
from sklearn.base import BaseEstimator
from utils import get_data_from_str, get_outlier_model, save_data_to_s3


def lambda_handler(event, context):
    """Lambda handler to ingest data, make prediction, and return prediction to client.
    """

    # Amazon API Gateway wraps input into a dictionary as the body
    # and local invocation uses raw input and does not wrap it
    if 'body' in event:
        data = get_data_from_str(event['body'])
    else:
        data = get_data_from_str(event)

    response = {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
        }
    }

    if not data:
        response['body'] = json.dumps("Malformed data input.")
        return response

    save_success = save_data_to_s3(data)
    model = get_outlier_model(data[0])

    if model:
        pred = predict(model, np.array(data[1:3]))
        response['body'] = json.dumps(pred)
    else:
        response['body'] = json.dumps("Model not retrieved from S3 successfully.")

    return response


def predict(model: BaseEstimator, sample: list) -> int:
    """Make a prediction taken from a Json event body.

    Returns:
    A -1 indicating an outlier or 1 indicating a normal value.
    """

    # Need to reshape a single sample because input needs to be 2d
    result = model.predict(sample.reshape(1, -1))
    return int(result[0])