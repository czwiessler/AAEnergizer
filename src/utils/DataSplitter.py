import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(dataset_path, output_dir, train_size=0.7, val_size=0.2, test_size=0.1, random_state=42):
    """
    Splits a dataset from a CSV file into train, validation, and test sets.

    Parameters:
        dataset_path (str): Path to the input CSV file.
        output_dir (str): Directory to save the split datasets.
        train_size (float): Proportion of the data to be used for training.
        val_size (float): Proportion of the data to be used for validation.
        test_size (float): Proportion of the data to be used for testing.
        random_state (int): Random state for reproducibility.

    Returns:
        None
    """
    if round(train_size + val_size + test_size, 4) != 1.0:
        raise ValueError("train_size, val_size, and test_size must sum to 1.0")

    # Load the dataset
    df = pd.read_csv(dataset_path)

    # Split the data into train and temp sets
    train_df, temp_df = train_test_split(df, test_size=(val_size + test_size), random_state=random_state)

    # Calculate proportions for validation and test splits
    val_proportion = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(temp_df, test_size=1 - val_proportion, random_state=random_state)

    # Save the splits to CSV
    train_df.to_csv(f"{output_dir}/train_set.csv", index=False)
    val_df.to_csv(f"{output_dir}/val_set.csv", index=False)
    test_df.to_csv(f"{output_dir}/test_set.csv", index=False)

    print(f"Dataset split completed:")
    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")

# Example usage
split_dataset("data/processed/charging_sessions_cleaned.csv", "data/processed/split/")
