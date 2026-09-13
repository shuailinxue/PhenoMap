import torch
import torch.nn as nn
from transformers import AutoModel

class KEEPVisionEncoder(nn.Module):

    def __init__(self, config=None, model_path="Astaxanthin/KEEP"):
        super().__init__()

        if config is not None and 'model' in config:
            self.model_path = config['model'].get('keep_model_path', model_path)
        else:
            self.model_path = model_path

        print(f"Loading official KEEP vision backbone from: {self.model_path}")

        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.vision_dim = 768

    @property
    def output_dim(self):
        return self.vision_dim

    def forward(self, images):
        if not isinstance(images, torch.Tensor):
            raise ValueError("Expected torch.Tensor input")

        self.model.eval()

        if len(images.shape) == 5:  # [B, N, 3, H, W]
            B, N, C, H, W = images.shape
            images_flat = images.reshape(B * N, C, H, W)

            with torch.no_grad():
                features = self.model.encode_image(images_flat)

            features = features.reshape(B, N, -1)
        elif len(images.shape) == 4:  # [B, 3, H, W]
            with torch.no_grad():
                features = self.model.encode_image(images)
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got {len(images.shape)}D")

        return features
