import requests
import pandas as pd
from datasets import load_dataset

def load_dataset(dataset_url):
    if dataset_url.startswith("http"):
        response = requests.get(dataset_url)
        with open("temp_dataset.csv", "wb") as f:
            f.write(response.content)
        dataset = load_dataset("csv", data_files="temp_dataset.csv")
    else:
        dataset = load_dataset("csv", data_files=dataset_url)
    return dataset
