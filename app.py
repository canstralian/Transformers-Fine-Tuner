import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import os

# Constants for default values
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 5e-5

def load_and_tokenize_dataset(dataset_url, file, tokenizer):
    """
    Load and tokenize the dataset.

    Args:
        dataset_url (str): URL of the dataset.
        file (file): Uploaded dataset file.
        tokenizer (AutoTokenizer): Tokenizer for the model.

    Returns:
        dataset (Dataset): Tokenized dataset.
        error_message (str): Error message if any.
    """
    if dataset_url:
        dataset = load_dataset(dataset_url)
    elif file:
        dataset = load_dataset("csv", data_files={"train": file.name})
    else:
        return None, "Please provide a dataset URL or upload a file."

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    dataset = dataset.map(tokenize_function, batched=True)
    return dataset, None

def initialize_trainer(model, dataset, tokenizer, epochs, batch_size, learning_rate):
    """
    Initialize the Trainer.

    Args:
        model (AutoModelForSequenceClassification): Model to be fine-tuned.
        dataset (Dataset): Tokenized dataset.
        tokenizer (AutoTokenizer): Tokenizer for the model.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for optimizer.

    Returns:
        trainer (Trainer): Initialized Trainer object.
    """
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",       # Save model at the end of each epoch
        logging_strategy="epoch",    # Log metrics at the end of each epoch
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        push_to_hub=False,
        report_to="all"  # Report to all available logging platforms
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],  # Ensure you have a test split
        tokenizer=tokenizer,
    )
    return trainer

# Function to fine-tune model
def fine_tune(model_name, dataset_url, file, epochs, batch_size, learning_rate):
    """
    Fine-tune a pre-trained transformer model on a custom dataset.

    Args:
        model_name (str): Name of the pre-trained model.
        dataset_url (str): URL of the dataset.
        file (file): Uploaded dataset file.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for optimizer.

    Returns:
        str: Result message.
    """
    try:
        # Load model & tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load and tokenize dataset
        dataset, error_message = load_and_tokenize_dataset(dataset_url, file, tokenizer)
        if error_message:
            return error_message

        # Initialize Trainer
        trainer = initialize_trainer(model, dataset, tokenizer, epochs, batch_size, learning_rate)

        # Start training
        trainer.train()

        return "Fine-tuning complete."

    except Exception as e:
        return f"An error occurred: {e}"

# Gradio interface
iface = gr.Interface(
    fn=fine_tune,
    inputs=[
        gr.Textbox(
            label="Model Name",
            placeholder="e.g., bert-base-uncased",
            info="Enter the name of the pre-trained model you want to fine-tune.",
        ),
        gr.Textbox(
            label="Dataset URL (optional)",
            placeholder="Enter dataset URL",
            info="Provide a URL to a dataset in CSV format.",
        ),
        gr.File(
            label="Upload Dataset (optional)",
            file_types=[".csv"],
            info="Upload a CSV file containing your dataset.",
        ),
        gr.Slider(
            minimum=1,
            maximum=10,
            step=1,
            value=3,
            label="Epochs",
            info="Number of times to iterate over the entire training dataset.",
        ),
        gr.Slider(
            minimum=1,
            maximum=64,
            step=1,
            value=8,
            label="Batch Size",
            info="Number of samples per batch during training.",
        ),
        gr.Slider(
            minimum=1e-6,
            maximum=1e-2,
            step=1e-6,
            value=5e-5,
            label="Learning Rate",
            info="Learning rate for the optimizer.",
        ),
    ],
    outputs="text",
    live=True,
    title="Transformers Fine Tuner",
    description="A user-friendly Gradio interface for fine-tuning pre-trained transformer models on your dataset.",
    theme="huggingface",
)

if __name__ == "__main__":
    iface.launch()
