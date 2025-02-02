---

# Transformers Fine Tuner

Transformers Fine Tuner is a user-friendly Gradio interface that enables seamless fine-tuning of pre-trained transformer models on custom datasets.

## Features

- **Easy Dataset Integration:** Load datasets via URLs or direct file uploads.
- **Model Selection:** Choose from a variety of pre-trained transformer models.
- **Customizable Training Parameters:** Adjust epochs, batch size, and learning rate to suit your needs.
- **Real-time Monitoring:** Track training progress and performance metrics.

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/canstralian/Transformers-Fine-Tuner.git
    cd Transformers-Fine-Tuner
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To launch the Gradio interface, run the following command:
```bash
python app.py
```

## Example

1. Enter the URL of your dataset.
2. Select a pre-trained transformer model from the dropdown.
3. Adjust the training parameters such as epochs, batch size, and learning rate.
4. Click the "Submit" button to start the fine-tuning process.

## File Structure

- `app.py`: Main script to launch the Gradio interface.
- `data/preprocess.py`: Script to load and preprocess datasets.
- `.github/workflows/python-app.yml`: GitHub Actions workflow for CI/CD pipeline.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

This project uses the following libraries and frameworks:
- [Gradio](https://gradio.app/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Pandas](https://pandas.pydata.org/)

## Contact

For any inquiries or support, please contact the repository owner at [canstralian](https://github.com/canstralian).

---

