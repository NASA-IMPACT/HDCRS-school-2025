import os

NO_DATA = -1
NO_DATA_FLOAT = 0.0001
PERCENTILES = (0.1, 99.9)

CROP_SIZE = (224, 224)
SPLITS = ['training', 'validation', 'test']

LAYERS = ['HLSS30', 'HLSL30']
DEFAULT_BASE_PATH = '/opt/ml/'
BUCKET_NAME = os.environ['BUCKET_NAME']
MODEL_PATH = "models/{model_name}"
