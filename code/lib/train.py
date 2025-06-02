from __future__ import absolute_import

import argparse
import os

from terratorch.cli_tools import build_lightning_cli
from glob import glob

from utils import assumed_role_session, download_data, save_model_artifacts

ROLE_ARN = os.environ.get('ROLE_ARN')
ROLE_NAME = os.environ.get('ROLE_NAME')


def train():
    config_file = os.environ.get('CONFIG_FILE')
    print(f'\n config file: {config_file}')

    # download and prepare data for training:
    for split in ['training', 'validation', 'test', 'configs']:
        download_data(os.environ.get('S3_URL'), split)

    # set random seeds
    lightning_cli = build_lightning_cli([
            "fit",
            "--config",
            config_file
        ],
        run=True
    )
    # At the end of the training loop, we have to save model artifacts.
    print("Training done, saving model")
    model_dir = os.environ["MODEL_DIR"]
    session = assumed_role_session()
    s3_connection = session.resource('s3')
    save_model_artifacts(s3_connection, model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    train()
