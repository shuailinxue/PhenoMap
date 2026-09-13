import torch
import torch.nn as nn
import json
import os
import sys
import types

import logging
logging.getLogger("scGPT").setLevel(logging.WARNING)
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

class ScGPTEncoder(nn.Module):

    def __init__(self, config=None, model_dir=None):
        super().__init__()

        if config is not None and 'model' in config:
            self.model_dir = config['model'].get('scgpt_model_dir', model_dir)
        else:
            self.model_dir = model_dir

        if not self.model_dir or not os.path.exists(self.model_dir):
            print("Downloading the scGPT backbone from Hugging Face Hub...")
            try:
                from huggingface_hub import snapshot_download
                repo_id = "VirtualCell2025/scGPT_human"
                self.model_dir = snapshot_download(repo_id=repo_id)
                print(f"Cached scGPT backbone at {self.model_dir}")
            except ImportError:
                raise ImportError("Install huggingface_hub to download the scGPT backbone")
        else:
            print(f"Loading genuine scGPT backbone from local: {self.model_dir}")

        try:
            sys.modules.setdefault("scgpt.tasks", types.ModuleType("scgpt.tasks"))
            from scgpt.model import TransformerModel
            from scgpt.tokenizer.gene_tokenizer import GeneVocab
            logging.getLogger("scGPT").setLevel(logging.WARNING)
        except ImportError:
            raise ImportError("Install scgpt to use the gene-expression encoder")

        vocab_file = os.path.join(self.model_dir, "vocab.json")
        model_config_file = os.path.join(self.model_dir, "args.json")
        model_file = os.path.join(self.model_dir, "best_model.pt")

        if not os.path.exists(model_file):
            raise FileNotFoundError(f"scGPT checkpoint not found: {model_file}")

        self.vocab = GeneVocab.from_file(vocab_file)
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)

        self.model = TransformerModel(
            ntoken=len(self.vocab),
            d_model=model_configs["embsize"],
            nhead=model_configs["nheads"],
            d_hid=model_configs["d_hid"],
            nlayers=model_configs["nlayers"],
            nlayers_cls=model_configs["n_layers_cls"],
            n_cls=1,
            vocab=self.vocab,
            dropout=model_configs["dropout"],
            pad_token=model_configs["pad_token"],
            pad_value=model_configs["pad_value"],
            do_mvc=model_configs.get("do_mvc", False),
            do_dab=model_configs.get("do_dab", False),
            use_batch_labels=model_configs.get("use_batch_labels", False),
            domain_spec_batchnorm=model_configs.get("domain_spec_batchnorm", False),
        )

        from scgpt.utils import load_pretrained
        pretrained_dict = torch.load(model_file, map_location="cpu")
        load_pretrained(self.model, pretrained_dict, strict=False)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.embed_dim = model_configs["embsize"]

    @property
    def output_dim(self):
        return self.embed_dim

    def extract_gene_features(self, gene_ids, values):
        self.model.eval()

        original_shape = gene_ids.shape
        if len(original_shape) == 3:
            B, N, seq_len = original_shape
            gene_ids_flat = gene_ids.reshape(B * N, seq_len)
            values_flat = values.reshape(B * N, seq_len)
        else:
            gene_ids_flat = gene_ids
            values_flat = values

        with torch.no_grad():
            src_key_padding_mask = gene_ids_flat.eq(self.vocab["<pad>"])

            token_embeddings = self.model._encode(
                gene_ids_flat,
                values_flat,
                src_key_padding_mask=src_key_padding_mask
            )

            if isinstance(token_embeddings, dict):
                cell_embeddings = token_embeddings['cell_emb']
            else:
                valid_mask = (~src_key_padding_mask).unsqueeze(-1).float()  # [B, SeqLen, 1]

                masked_embeddings = token_embeddings * valid_mask

                sum_embeddings = masked_embeddings.sum(dim=1)  # [B, 512]

                valid_counts = valid_mask.sum(dim=1).clamp(min=1.0)  # [B, 512]

                cell_embeddings = sum_embeddings / valid_counts

        if len(original_shape) == 3:
            cell_embeddings = cell_embeddings.reshape(B, N, -1)

        return cell_embeddings
