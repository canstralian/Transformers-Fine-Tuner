import gradio as gr
from model.model import fine_tune
from data.preprocess import load_data, preprocess_data, save_processed_data

def prepare_and_train(model_name, dataset_path, epochs, batch_size, learning_rate):
    # Load and preprocess the dataset
    data = load_data(dataset_path)
    cleaned_data = preprocess_data(data)
    processed_data_path = 'data/processed/processed_dataset.csv'
    save_processed_data(cleaned_data, processed_data_path)

    # Proceed with model fine-tuning
    return fine_tune(model_name, dataset_url=None, file=processed_data_path, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)

iface = gr.Interface(
    fn=prepare_and_train,
    inputs=[
        gr.Textbox(label="Model Name", placeholder="e.g., bert-base-uncased"),
        gr.File(label="Upload Dataset"),
        gr.Number(label="Epochs", value=3),
        gr.Number(label="Batch Size", value=8),
        gr.Number(label="Learning Rate", value=5e-5),
    ],
    outputs="text",
    live=True,
)

if __name__ == "__main__":
    iface.launch()
