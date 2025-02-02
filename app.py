import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import os

# Function to fine-tune model
def fine_tune(model_name, dataset_url, file, epochs, batch_size, learning_rate):
    try:
        # Load dataset
        if dataset_url:
            dataset = load_dataset(dataset_url)
        elif file:
            dataset = load_dataset("csv", data_files={"train": file.name})
        else:
            return "Please provide a dataset URL or upload a file."

        # Load model & tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)

        dataset = dataset.map(tokenize_function, batched=True)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch
