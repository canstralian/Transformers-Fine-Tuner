import torch

def save_model(model, tokenizer, output_dir):
    """Save the model and tokenizer to the specified directory."""
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")

def load_model(model_class, tokenizer_class, model_dir):
    """Load the model and tokenizer from the specified directory."""
    model = model_class.from_pretrained(model_dir)
    tokenizer = tokenizer_class.from_pretrained(model_dir)
    return model, tokenizer
