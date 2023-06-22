import pickle
from functools import partial

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

from typing import Dict, Union, Tuple, Any


MODEL_PATH = './model/'

# Load tokenizer and backbone
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
backbone = AutoModel.from_pretrained(MODEL_PATH)

# Load projection head weights from file
projection_head = nn.Linear(312, 64)
with open(MODEL_PATH + 'projection_head.pkl', 'rb') as f:
    projection_head.load_state_dict(pickle.load(f))


def prepare_tokenized_text(text: str, tokenizer: AutoTokenizer, type_of_text: str = 'query') -> Dict[str, torch.Tensor]:
    """
    Preprocesses and tokenizes the input text.

    Args:
        text (str): The input text to be tokenized.
        tokenizer (transformers.AutoTokenizer): The tokenizer to use.
        type_of_text (str, optional): The type of text: query or item. Defaults to 'query'.

    Returns:
        dict: The tokenized text as a dictionary of tensors.
    """
    if isinstance(text, str):
        # Clean and format the text
        text = ' '.join(text.lower().split())

        # Add appropriate prefix based on type_of_text
        prefix = '[I]' if type_of_text == 'query' else '[Q]'
        text = prefix + text

    elif isinstance(text, list):
        # Clean and format the text
        text = [' '.join(t.lower().split()) for t in text]

        # Add appropriate prefix based on type_of_text
        prefix = '[I]' if type_of_text == 'query' else '[Q]'
        text = [prefix + t for t in text]

    # Tokenize the text
    return tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=64)


def embed_tokenized_text(tokenized_text: Dict[str, torch.Tensor], model: AutoModel, projection_head: nn.Module) -> torch.Tensor:
    """
    Embeds the tokenized text using a model with a projection head.

    Args:
        tokenized_text (dict): The tokenized text as a dictionary of tensors.
        model (transformers.AutoModel): The model to use for embedding.
        projection_head (torch.nn.Module): The projection head to apply.

    Returns:
        numpy.ndarray: The embedded text as a numpy array.
    """
    with torch.no_grad():
        # Pass tokenized_text to the model and retrieve the model output
        model_output = model(**{k: v.to(model.device) for k, v in tokenized_text.items()})
    
        # Extract the embeddings from the model output
        embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = nn.functional.normalize(embeddings)

        # Apply the projection head to obtain the final embeddings
        embeddings = projection_head(embeddings)
        embeddings = torch.nn.functional.normalize(embeddings)

    return embeddings.cpu().numpy()


# Create partial functions with pre-defined arguments
tokenizer_full = partial(prepare_tokenized_text, tokenizer=tokenizer)
model_full = partial(embed_tokenized_text, model=backbone, projection_head=projection_head)


def create_model() -> Tuple[partial, partial]:
    """
    Creates and returns a model consisting of the tokenizer and embedding function.

    Returns:
        tuple: A tuple containing the tokenizer function and the embedding function.
    """
    return (tokenizer_full, model_full)


if __name__ == '__main__':
    text = 'Hello World!'

    # create tokenizer and model
    tokenizer_full, model_full = create_model()

    # Tokenize the text
    tokenized_text = tokenizer_full(text)

    # Print the shape of the embeddings
    embeddings = model_full(tokenized_text)
    print('Shape:', embeddings.shape)
