import os
import rioxarray as rxr
import torch

from lib.utils import assumed_role_session
from lib.consts import BUCKET_NAME

from terratorch.registry import FULL_MODEL_REGISTRY
from terratorch.tasks.tiled_inference import tiled_inference

DOWNLOAD_PATH = '/opt/ml/data/downloads'

if not(os.path.exists(DOWNLOAD_PATH)):
    os.makedirs(DOWNLOAD_PATH)

MODEL = FULL_MODEL_REGISTRY.build(
    'terramind_v1_base_generate',
    modalities=['S2L2A'],
    output_modalities=['S1GRD', 'DEM', 'LULC', 'NDVI'],
    pretrained=True,
    standardize=True,
    timesteps=10,  # Number of diffusion steps
).to('cuda')

class InferGeneration:
    def __init__(self, s3_image_path):
        self.s3_image_path = s3_image_path
        self.device = 'cuda'
        self.model = MODEL
        self.download_image()

    def download_image(self):
        session = assumed_role_session()
        s3_connection = session.resource('s3')
        bucket = s3_connection.Bucket(BUCKET_NAME)
        filename = self.s3_image_path.split('/')[-1]
        self.file_path = f"{DOWNLOAD_PATH}/{filename}"
        if not(os.path.exists(self.file_path)):
            bucket.download_file(self.s3_image_path.replace(f's3://{BUCKET_NAME}/', ''), self.file_path)
        print(f"File downloaded from {self.s3_image_path} to {self.file_path}")


    # Define model forward for tiled inference
    def model_forward(self, x):
        # Run chained generation for all output modalities
        generated = self.model(x)

        # TerraTorch tiled inference expects a tensor output from model forward. We concatenate all generations along channel dimension.
        out = torch.concat([
            generated['S1GRD'],
            generated['DEM'],
            generated['LULC'],
            generated['NDVI']
        ], dim=1)

        return out


    def tiled_infer(self, reduce=True):
        data = rxr.open_rasterio(self.file_path).values
        if reduce:
            data = data[:, 500:1500] # Optionally reduce image size to speed up inference
        data_input = torch.tensor(data, dtype=torch.float, device=self.device).unsqueeze(0)
        pred = tiled_inference(self.model_forward, data_input, crop=256, stride=224, batch_size=8, verbose=True)
        pred = pred.squeeze(0)
        lulc = pred[3:13].argmax(0).tolist()
        s1grd = pred[0:2].tolist()
        dem = pred[2].tolist()
        ndvi = pred[-1].tolist()

        # Split up the stacked bands into each modality
        return {
            'S1GRD': s1grd,
            'DEM': dem,
            'LULC': lulc,  # Get class predictions from logits
            'NDVI': ndvi
        }
