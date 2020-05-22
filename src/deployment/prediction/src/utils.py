import boto3
import joblib
from botocore.exceptions import ClientError
from sklearn.base import BaseEstimator
from src.config import BUCKET_NAME, MODEL_EXTENSION, MODEL_FILENAME


def upload_model_to_s3(model: BaseEstimator, user: int, local_path='/tmp/', profile='default') -> bool:
    """Upload model file to S3 and save locally in `local_path`.

    Returns:
    True if successfully uploaded the file or False if a ClientError occurred.
    """

    dev = boto3.session.Session(profile_name=profile)
    S3 = dev.client('s3')
    bucket = BUCKET_NAME
    filename = MODEL_FILENAME + str(user) + MODEL_EXTENSION
    local_filename = local_path + filename
    model_filename = 'model/' + filename
    joblib.dump(model, local_filename)
    return upload_file_to_s3(local_filename, bucket, model_filename, S3)


def upload_file_to_s3(local_filename, bucket, model_filename, S3=boto3.client('s3')) -> bool:
    """Upload file to S3.

    Returns:
    True if successfully uploaded the file or False if a ClientError occurred.
    """

    try:
        S3.upload_file(local_filename, bucket, model_filename)
    except ClientError as e:
        print("Upload not a success with error:", e)
        return False
    return True


def upload_text_to_s3(text, bucket, s3_filename, S3=boto3.resource('s3')) -> bool:
    """Upload text to S3 object.

    Returns:
    True if successfully uploaded the file or False if a ClientError occurred.
    """

    try:
        S3.Object(bucket, s3_filename).put(Body=text)
    except ClientError as e:
        print("Upload not a success with error:", e)
        return False
    return True


def save_data_to_s3(data, bucket=BUCKET_NAME, S3=boto3.resource('s3')) -> bool:
    """Takes the inputs and saves them as a csv formatted string to s3.

    Path is formatted in s3 as shown below:
    user/year/month/day/hour/minute_second.txt

    Returns:
    True if successfully uploaded the file or False if an error occurred.
    """
    date_path = extract_date_path(data[-1])
    file_path = extract_file_path(data[-1]) + '.csv'
    full_path = 'data/' + str(data[0]) + '/' + date_path + file_path
    return upload_text_to_s3(stringify_list(data), bucket, full_path, S3)


def extract_date_path(string: str) -> str:
    # user/year/month/day/hour/
    return string.replace('-','/').replace(' ', '/').replace(':', '/')[:-5]

def extract_file_path(string: str) -> str:
    # user/year/month/day/hour/
    return string.replace(':','_')[-5:]


def stringify_list(data: list) -> str:
    """Turn list into csv formatted string."""

    return ','.join([str(val) for val in data])


def get_data_from_str(string: str) -> list:
    """Input a csv string representing the data and return a list that has each value properly
    casted. If string isn't properly formatted, then return an empty list.

    """

    split_vals = string.split(',')
    if len(split_vals) != 5:
        return []

    user_id = int(split_vals[0])
    hr = float(split_vals[1])
    rr = float(split_vals[2])
    inroom = bool(int(split_vals[3]))
    ts = split_vals[4]
    return [user_id, hr, rr, inroom, ts]


def get_outlier_model(user: int) -> BaseEstimator:
    """Get the outlier model from S3.

    Returns:
    Sklearn model for prediction.
    """

    S3 = boto3.client('s3')
    bucket = BUCKET_NAME
    filename = MODEL_FILENAME + str(user) + MODEL_EXTENSION
    model_filename = 'model/' + filename
    local_filename = '/tmp/' + filename
    try:
        S3.download_file(bucket, model_filename, local_filename)
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print("Object does not exist.")
        else:
            print("Unkwown error.")
        return None
    return joblib.load(local_filename)
