import pandas as pd
import os

def load_data(file_path):
    """Load dataset from the specified file path."""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or JSON file.")

def preprocess_data(df):
    """Perform data preprocessing steps."""
    # Example preprocessing: drop rows with missing values
    df_cleaned = df.dropna()
    # Additional preprocessing steps can be added here
    return df_cleaned

def save_processed_data(df, output_path):
    """Save the processed DataFrame to the specified path."""
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    raw_data_path = 'data/raw/dataset.csv'  # Replace with your raw data file path
    processed_data_path = 'data/processed/dataset_cleaned.csv'

    # Load raw data
    data = load_data(raw_data_path)

    # Preprocess data
    cleaned_data = preprocess_data(data)

    # Save processed data
    save_processed_data(cleaned_data, processed_data_path)
