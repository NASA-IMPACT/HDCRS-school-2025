import torch
import yaml
import numpy as np
import rasterio
from terratorch.cli_tools import LightningInferenceModel
from lib.consts import NO_DATA, NO_DATA_FLOAT

class Infer:
    def __init__(self, config, checkpoint):
        self.config_filename = config
        with open(self.config_filename) as config:
            self.config = yaml.safe_load(config)
        self.checkpoint_filename = checkpoint
        self.load_model()

    def load_model(self):
        inference_model = LightningInferenceModel.from_config(self.config_filename, self.checkpoint_filename)
        self.model = inference_model.model.to('cuda')

        # load model using terratorch
        self.model = self.model.eval()

    def preprocess(self, images, terramind=False):
        images_array = []
        profiles = []

        mean = []
        std = []
        if terramind:
            modality = 'S2L1C'
            with rasterio.open(images[0]) as raster_file:
                if raster_file.count == 12:
                    modality = 'S2L1A'
            mean = torch.tensor(self.config['data']['init_args']['means'][modality]).view(-1, 1, 1)
            std = torch.tensor(self.config['data']['init_args']['stds'][modality]).view(-1, 1, 1)
        else:
            mean = torch.tensor(self.config['data']['init_args']['means']).view(-1, 1, 1)
            std = torch.tensor(self.config['data']['init_args']['stds']).view(-1, 1, 1)

        for image in images:
            with rasterio.open(image) as raster_file:
                image = raster_file.read()
                image = np.where(image == NO_DATA, NO_DATA_FLOAT, image)
                image = torch.from_numpy(image)
                image = (image - mean) / std
                images_array.append(image)
                profiles.append(raster_file.profile)
                raster_file.close()
        # Example processing function to simulate the pipeline
        imgs_tensor = torch.from_numpy(np.asarray(images_array))  # Assuming input_array is of type np.float32
        imgs_tensor = imgs_tensor.float()

        # increase dimensions to match input size
        processed_images = imgs_tensor
        if not(terramind):
            processed_images = imgs_tensor.unsqueeze(2)
        return processed_images, profiles

    def infer(self, images, terramind=False):
        """
        Infer on provided images
        Args:
            images (list): List of images
        """
        # forward the model
        with torch.no_grad():
            images, profiles = self.preprocess(images, terramind=terramind)
            result = self.model(images.to('cuda'))
            predicted_masks = list()
            results = result.output.detach().cpu()
            for index, mask in enumerate(results):
                output = mask.cpu()  # [n_segmentation_class, 224, 224]
                if self.config['data']['init_args']['num_classes'] == 1:
                    updated_mask = torch.sigmoid(output.clone()).squeeze(0)
                    predicted_mask = (updated_mask > self.config.get('threshold', 0.5)).int()
                else:
                    predicted_mask = mask.argmax(dim=0)
                    img_size = profiles[index]['width']
                    predicted_mask = torch.nn.functional.interpolate(
                            predicted_mask.unsqueeze(0).float(),
                            size=img_size,
                            mode="nearest"
                        )
                predicted_masks.append(predicted_mask)
        return predicted_masks, profiles
