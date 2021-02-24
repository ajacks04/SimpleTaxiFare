import pandas as pd
import numpy as np
import joblib
from sklearn import linear_model
from google.cloud import storage


GCP_BUCKET_PATH = 'gs://le-wagon-bootcamp-305807/train_1k.csv'
BUCKET_NAME='le-wagon-bootcamp-305807'



def get_data(nrows=100):
    """method used in order to get the training data (or a portion of it) from google cloud bucket"""
    df = pd.read_csv(GCP_BUCKET_PATH, nrows=nrows, error_bad_lines=False)
    return df


def compute_distance(df,
                         start_lat="pickup_latitude",
                         start_lon="pickup_longitude",
                         end_lat="dropoff_latitude",
                         end_lon="dropoff_longitude"):
    """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees).
        Vectorized version of the haversine distance for pandas df
        Computes distance in kms
    """

    lat_1_rad, lon_1_rad = np.radians(df[start_lat].astype(float)),\
        np.radians(df[start_lon].astype(float))
    lat_2_rad, lon_2_rad = np.radians(df[end_lat].astype(float)),\
        np.radians(df[end_lon].astype(float))
    dlon = lon_2_rad - lon_1_rad
    dlat = lat_2_rad - lat_1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_1_rad) * np.cos(lat_2_rad) *\
        np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c



def preprocess(df):
    """method that pre-processes the data"""
    df["distance"] = compute_distance(df)
    X_train = df[["distance"]]
    y_train = df["fare_amount"]
    return X_train, y_train


def train_model(X_train, y_train):
    """method that trains the model"""
    rgs = linear_model.Lasso(alpha=0.1)
    rgs.fit(X_train, y_train)
    return rgs


def save_model(reg):
    """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
    HINTS : use joblib library and google-cloud-storage"""
    filename = 'finalized_model.sav'
    joblib.dump(reg, filename)
    print("saved model.joblib locally")

    client = storage.Client()
    bucket = client.get_bucket('le-wagon-bootcamp-305807')
    blob = bucket.blob('models/taxi_fare_model/finalized_model.sav')
    blob.upload_from_filename(filename)
    print("uploaded model.joblib to gcp cloud storage")


if __name__ == '__main__':
    df = get_data()
    X_train, y_train = preprocess(df)
    clf = train_model(X_train, y_train)
    save_model(clf)