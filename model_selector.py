from transformers import HUGGINGFACE_HUB_NAME, HUGGINGFACE_HUB_MODEL

def get_model_list():
    return [
        "bert-base-uncased",
        "distilbert-base-uncased",
        "roberta-base",
        "gpt2"
    ]
