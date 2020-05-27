import boto3
import joblib
from botocore.exceptions import ClientError
from sklearn.base import BaseEstimator
from src.config import BUCKET_NAME, MODEL_EXTENSION, MODEL_FILENAME, SCALER_FILENAME


def upload_model_to_s3(model: BaseEstimator, user: int, model_name: str, local_path='/tmp/',
                       profile='default') -> bool:
    """Upload model file to S3 and save locally in `local_path`.

    Returns:
    True if successfully uploaded the file or False if a ClientError occurred.
    """

    dev = boto3.session.Session(profile_name=profile)
    S3 = dev.client('s3')
    bucket = BUCKET_NAME
    filename = model_name + str(user) + MODEL_EXTENSION
    local_filename = local_path + filename
    s3_filepath = 'model/' + filename
    joblib.dump(model, local_filename)
    return upload_file_to_s3(local_filename, bucket, s3_filepath, S3)


def upload_file_to_s3(local_filename, bucket, s3_filepath, S3=boto3.client('s3')) -> bool:
    """Upload file to S3.

    Returns:
    True if successfully uploaded the file or False if a ClientError occurred.
    """

    try:
        S3.upload_file(local_filename, bucket, s3_filepath)
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
    date_path = extract_date_path(data['ts'])
    file_path = extract_file_path(data['ts']) + '.csv'
    full_path = 'data/' + str(data['user_id']) + '/' + date_path + file_path
    return upload_text_to_s3(stringify_list(data), bucket, full_path, S3)


def extract_date_path(string: str) -> str:
    """Get date path from time string field."""
    # user/year/month/day/hour/
    return string.replace('-', '/').replace(' ', '/').replace(':', '/')[:-5]


def extract_file_path(string: str) -> str:
    """Get file path from time string field."""
    # minute_second
    return string.replace(':', '_')[-5:]


def stringify_list(data: list) -> str:
    """Turn list into csv formatted string."""

    return ','.join([str(val) for val in data])


def get_data_from_str(string: str) -> list:
    """Input a csv string representing the data and return a list that has each value properly
    casted. If string isn't properly formatted, then return an empty list.

    """

    data_dict = {}
    split_vals = string.split(',')
    if len(split_vals) != 5:
        return data_dict

    data_dict['user_id'] = int(split_vals[0])
    data_dict['hr'] = float(split_vals[1])
    data_dict['rr'] = float(split_vals[2])
    data_dict['inroom'] = bool(int(split_vals[3]))
    data_dict['ts'] = split_vals[4]
    return data_dict


def get_outlier_model(user: int) -> BaseEstimator:
    """Get the outlier model from S3.

    Returns:
    Sklearn model for prediction.
    """

    bucket = BUCKET_NAME
    filename = MODEL_FILENAME + str(user) + MODEL_EXTENSION
    s3_filepath = 'model/' + filename
    local_filename = '/tmp/' + filename
    if download_from_S3(bucket, s3_filepath, local_filename):
        return joblib.load(local_filename)
    return None


def get_scaler(user: int) -> BaseEstimator:
    """Get scaler from S3.

    Returns:
    Sklearn Scaler for transforming features.
    """

    bucket = BUCKET_NAME
    filename = SCALER_FILENAME + str(user) + MODEL_EXTENSION
    s3_filepath = 'model/' + filename
    local_filename = '/tmp/' + filename
    if download_from_S3(bucket, s3_filepath, local_filename):
        return joblib.load(local_filename)
    return None


def download_from_S3(bucket, s3_filename, local_filename, S3=boto3.client('s3')) -> str:
    """Download file from S3."""
    try:
        S3.download_file(bucket, s3_filename, local_filename)
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print("Object does not exist.")
        else:
            print("Unkwown error.")
        return None
    return local_filename
