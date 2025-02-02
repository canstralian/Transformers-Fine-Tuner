---
title: Transformers Fine Tuner
emoji: 🔥
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.14.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: A Gradio interface
---

![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-Apache%202.0-blue)
![Last Commit](https://img.shields.io/github/last-commit/Canstralian/transformers-fine-tuner)
![Issues](https://img.shields.io/github/issues/Canstralian/transformers-fine-tuner)
![Pull Requests](https://img.shields.io/github/issues-pr/Canstralian/transformers-fine-tuner)
![Contributors](https://img.shields.io/github/contributors/Canstralian/transformers-fine-tuner)

# Transformers Fine Tuner

🔥 **Transformers Fine Tuner** is a user-friendly Gradio interface that enables seamless fine-tuning of pre-trained transformer models on custom datasets. This tool facilitates efficient model a...

## Features

- **Easy Dataset Integration**: Load datasets via URLs or direct file uploads.
- **Model Selection**: Choose from a variety of pre-trained transformer models.
- **Customizable Training Parameters**: Adjust epochs, batch size, and learning rate to suit your needs.
- **Real-time Monitoring**: Track training progress and performance metrics.

## Getting Started

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/canstralian/Transformers-Fine-Tuner.git
    cd transformers-fine-tuner
    ```

2. **Set Up a Virtual Environment** (optional but recommended):
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. **Install Dependencies**:
    Ensure you have Python 3.10 or higher. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Application**:
    ```bash
    python app.py
    ```
    Access the interface at `http://localhost:7860/`.

## Usage

- **Model Name**: Enter the name of the pre-trained model you wish to fine-tune (e.g., `bert-base-uncased`).
- **Dataset URL**: Provide a URL to your dataset.
- **Upload Dataset**: Alternatively, upload a dataset file directly.
- **Number of Epochs**: Set the number of training epochs.
- **Learning Rate**: Specify the learning rate for training.
- **Batch Size**: Define the batch size for training.

After configuring the parameters, click **Submit** to start the fine-tuning process. Monitor the training progress and performance metrics in real-time.

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Gradio](https://gradio.app/)
- [Datasets](https://huggingface.co/docs/datasets/)
