import gradio as gr
from fine_tuner import fine_tune_model
from model_selector import get_model_list
from utils import load_dataset

def train_model(dataset_url, model_name, epochs, batch_size, learning_rate):
    dataset = load_dataset(dataset_url)
    metrics = fine_tune_model(dataset, model_name, epochs, batch_size, learning_rate)
    return metrics

def main():
    model_options = get_model_list()
    interface = gr.Interface(
        fn=train_model,
        inputs=[
            gr.inputs.Textbox(label="Dataset URL"),
            gr.inputs.Dropdown(choices=model_options, label="Select Model"),
            gr.inputs.Slider(minimum=1, maximum=10, default=3, label="Epochs"),
            gr.inputs.Slider(minimum=1, maximum=64, default=16, label="Batch Size"),
            gr.inputs.Slider(minimum=1e-5, maximum=1e-1, step=1e-5, default=1e-4, label="Learning Rate")
        ],
        outputs="json",
        title="Transformers Fine Tuner",
        description="Fine-tune pre-trained transformer models on custom datasets."
    )
    interface.launch()

if __name__ == "__main__":
    main()
