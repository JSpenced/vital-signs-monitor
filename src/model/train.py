import sys
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import train_test_split
sys.path.append('../deployment/prediction/src/')
from utils import upload_model_to_s3
from config import MODEL_FILENAME, MODEL_EXTENSION, BUCKET_NAME, AWS_PROFILE


df = pd.read_csv('../../data/raw/data_2020_05.csv', parse_dates=['ts'])
user_ids = df.user_id.unique()

min_hr = 1  # Values below this ignored
min_rr = 3  # Values below this ignored
for user in user_ids:
    X = df[(df.in_room == True) &
           (df.user_id == user) &
           (df.hr > min_hr) &
           (df.rr > min_rr)][['hr', 'rr']]

    X_tr, X_va = train_test_split(X, test_size=0.2, shuffle=False)

    # Fit Gaussian to data to detect outliers
    el = EllipticEnvelope(contamination=0.12)
    el.fit(X_tr)

    savepath = '../../data/models/'
    upload_model_to_s3(el, user, savepath, profile=AWS_PROFILE)
