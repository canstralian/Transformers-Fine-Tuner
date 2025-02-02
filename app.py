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

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],  # Ensure you have a test split
            tokenizer=tokenizer,
        )

        # Start training
        trainer.train()

        return "Fine-tuning complete."

    except Exception as e:
        return f"An error occurred: {e}"

# Gradio interface
iface = gr.Interface(
    fn=fine_tune,
    inputs=[
        gr.Textbox(label="Model Name", placeholder="e.g., bert-base-uncased"),
        gr.Textbox(label="Dataset URL (optional)"),
        gr.File(label="Upload Dataset (optional)"),
        gr.Number(label="Epochs", value=3),
        gr.Number(label="Batch Size", value=8),
        gr.Number(label="Learning Rate", value=5e-5),
    ],
    outputs="text",
    live=True,
)

if __name__ == "__main__":
    iface.launch()
