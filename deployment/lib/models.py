from functools import partial

import torch

from transformers import AutoTokenizer, AutoModel
from transformers import logging
logging.set_verbosity_error()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_name = 'cointegrated/rubert-tiny2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)


def embed_bert_cls(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy() # (312, )

full_model = partial(embed_bert_cls, model=model, tokenizer=tokenizer)


# tokenizer = AutoTokenizer.from_pretrained('cointegrated/LaBSE-en-ru')
# model = AutoModel.from_pretrained('cointegrated/LaBSE-en-ru').to(device)

# def embed_labse_cls(text, model, tokenizer):
#     t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
#     with torch.no_grad():
#         model_output = model(**{k: v.to(model.device) for k, v in t.items()})
#     embeddings = model_output.pooler_output
#     embeddings = torch.nn.functional.normalize(embeddings)
#     return embeddings[0].cpu().numpy()

# full_model = partial(embed_labse_cls, model=model, tokenizer=tokenizer)


def create_model():
    return full_model
