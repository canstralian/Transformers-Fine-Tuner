import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

def fine_tune_model(dataset, model_name, epochs, batch_size, learning_rate):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_dir='./logs',
        logging_steps=10,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
    )
    trainer.train()
    return {"status": "Training complete"}
