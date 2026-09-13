import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class KEEPTextEncoder(nn.Module):

    def __init__(self, config=None, model_path="Astaxanthin/KEEP"):
        super().__init__()

        if config is not None and 'model' in config:
            self.model_path = config['model'].get('keep_model_path', model_path)
        else:
            self.model_path = model_path

        print(f"Loading official KEEP text backbone from: {self.model_path}")

        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.text_dim = 768

    @property
    def output_dim(self):
        return self.text_dim

    def forward(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        self.model.eval()

        device = next(self.model.parameters()).device

        with torch.no_grad():
            token_input = self.tokenizer(
                texts,
                max_length=256,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(device)

            features = self.model.encode_text(token_input)

        return features
