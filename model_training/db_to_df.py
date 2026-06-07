# db_to_df.py

import sqlite3
import pandas as pd
import json

def load_data_from_db(db_path='landmarks.db'):
    """
    Loads the landmarks data from a SQLite database into a Pandas DataFrame.
    Returns a DataFrame with columns: [id, file_path, split, label, landmarks].
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM landmarks", conn)
    conn.close()

    # Convert 'landmarks' from JSON string to Python list of (x, y, z) tuples
    def parse_landmarks(x):
        if x is None:
            return None
        return json.loads(x)

    df['landmarks'] = df['landmarks'].apply(parse_landmarks)
    return df

def prepare_dataframe(db_path='landmarks.db'):
    """
    1) Loads data from SQLite.
    2) Removes rows with null landmarks except those labeled 'nothing'.
    3) Separates data by split ('asl_alphabet_train' vs 'asl_alphabet_test').
    4) From the train subset, randomly samples 5% of each label and moves them to the test subset.
    5) Prints the final sizes of train and test datasets.
    Returns (df_train, df_test) as two DataFrames.
    """
    df = load_data_from_db(db_path)

    # Keep 'nothing' rows even if landmarks are None, but drop other null landmarks
    df = df[(df['label'] == 'nothing') | (df['landmarks'].notnull())]

    # Separate train vs. test based on 'split'
    df_train = df[df['split'] == 'asl_alphabet_train'].copy()
    df_test = df[df['split'] == 'asl_alphabet_test'].copy()

    # Sampling 5% per label
    new_df_train = pd.DataFrame(columns=df_train.columns)
    new_df_test = df_test.copy()

    for label in df_train['label'].unique():
        label_df = df_train[df_train['label'] == label]
        n_to_test = int(len(label_df) * 0.05)

        if n_to_test == 0:
            new_df_train = pd.concat([new_df_train, label_df], ignore_index=True)
        else:
            label_test_sample = label_df.sample(n=n_to_test, random_state=42)
            label_train_remaining = label_df.drop(label_test_sample.index)

            label_test_sample['split'] = 'asl_alphabet_test'

            new_df_train = pd.concat([new_df_train, label_train_remaining], ignore_index=True)
            new_df_test = pd.concat([new_df_test, label_test_sample], ignore_index=True)

    # Reset indexes
    new_df_train.reset_index(drop=True, inplace=True)
    new_df_test.reset_index(drop=True, inplace=True)

    # Print dataset sizes
    print(f"Train dataset size: {len(new_df_train)}")
    print(f"Test dataset size: {len(new_df_test)}")

    return new_df_train, new_df_test
