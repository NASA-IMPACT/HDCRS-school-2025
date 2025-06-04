import boto3
import os
from os import path
from glob import glob
from consts import BUCKET_NAME

MODEL_PATH = "models/{model_name}"


def assumed_role_session():
    # Assume the "notebookAccessRole" role we created using AWS CDK.
    client = boto3.client('sts')
    return boto3.session.Session()


def download_data(data, split):
    split_folder = f"/opt/ml/data/{split}"
    if not (os.path.exists(split_folder)):
        os.makedirs(split_folder)
    session = assumed_role_session()
    s3_connection = session.resource('s3')
    splits = data.split('/')
    bucket = s3_connection.Bucket(BUCKET_NAME)
    objects = list(bucket.objects.filter(Prefix="/".join(splits[3:] + [split])))
    print("Downloading files:", data, split)
    for iter_object in objects:
        splits = iter_object.key.split('/')
        if splits[-1]:
            filename = f"{split_folder}/{splits[-1]}"
            bucket.download_file(iter_object.key, filename)
    print("Finished downloading files.")


def save_model_artifacts(s3_connection, model_artifacts_path):
    if path.exists(model_artifacts_path):
        print(glob(f"{model_artifacts_path}"))
        for model_file in glob(f"{model_artifacts_path}/best-epoch*"):
            model_name = model_file.split('/')[-1]
            model_name = os.environ.get('MODEL_NAME', model_name)
            model_name = MODEL_PATH.format(model_name=model_name)
            print(f"Uploading model to s3: s3://{BUCKET_NAME}/{model_name}")
            s3_connection.meta.client.upload_file(model_file, BUCKET_NAME, model_name)
