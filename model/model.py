import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

def fine_tune(model_name, dataset_url=None, file=None, epochs=3, batch_size=8, learning_rate=5e-5):
    try:
        # Load dataset
        if dataset_url:
            dataset = load_dataset(dataset_url)
        elif file:
            dataset = load_dataset("csv", data_files={"train": file.name})
        else:
            raise ValueError("Please provide a dataset URL or upload a file.")

        # Load model & tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)

        dataset = dataset.map(tokenize_function, batched=True)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            push_to_hub=False,
            report_to="all"
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=tokenizer,
        )

        # Start training
        trainer.train()

        return "Fine-tuning complete."

    except Exception as e:
        return f"An error occurred: {e}"
