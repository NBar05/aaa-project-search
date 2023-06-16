import __main__

import torch
from torch import nn

from transformers import AutoTokenizer, AutoModel
from transformers import logging
logging.set_verbosity_error()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ProjectorModel(nn.Module):
    def __init__(self, model_name: str = 'cointegrated/rubert-tiny2', final_emb_size: int = 32):
        super().__init__()

        self.model_name = model_name
        self.final_emb_size = final_emb_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.backbone = AutoModel.from_pretrained(self.model_name, output_hidden_states=True).to(device)

        for n, p in self.backbone.named_parameters():
            p.requires_grad = False

        self.initial_emd_size = 312

        self.projection_head = nn.Sequential(
            nn.Linear(self.initial_emd_size, self.final_emb_size, device=device),            
        )

    def backbone_forward(self, text):
        t = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        model_output = self.backbone(**{k: v.to(self.backbone.device) for k, v in t.items()})

        embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = nn.functional.normalize(embeddings)

        return embeddings

    def forward(self, text):
        embeddings = self.backbone_forward(text)

        compressed_embeddings = self.projection_head(embeddings)
        compressed_embeddings = nn.functional.normalize(compressed_embeddings)

        return compressed_embeddings[0] if len(compressed_embeddings) == 1 else compressed_embeddings


setattr(__main__, 'ProjectorModel', ProjectorModel)

full_model = ProjectorModel(model_name='./model_tuned/', final_emb_size=64)
full_model = torch.load('./model_tuned/model_tuned_full.pkl', map_location=device)


def create_model():
    return full_model
