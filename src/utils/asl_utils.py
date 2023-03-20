import json
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np

import torch

### utils tool
def flatten_l_o_l(nested_list):
    """Flatten a list of lists into a single list.

    Args:
        nested_list (list): 
            – A list of lists (or iterables) to be flattened.

    Returns:
        list: A flattened list containing all items from the input list of lists.
    """
    return [item for sublist in nested_list for item in sublist]

def print_ln(symbol="-", line_len=110, newline_before=False, newline_after=False):
    """Print a horizontal line of a specified length and symbol.

    Args:
        symbol (str, optional): 
            – The symbol to use for the horizontal line
        line_len (int, optional): 
            – The length of the horizontal line in characters
        newline_before (bool, optional): 
            – Whether to print a newline character before the line
        newline_after (bool, optional): 
            – Whether to print a newline character after the line
    """
    if newline_before: print();
    print(symbol * line_len)
    if newline_after: print();
    
def read_json_file(file_path):
    """Read a JSON file and parse it into a Python object.

    Args:
        file_path (str): The path to the JSON file to read.

    Returns:
        dict: A dictionary object representing the JSON data.
        
    Raises:
        FileNotFoundError: If the specified file path does not exist.
        ValueError: If the specified file path does not contain valid JSON data.
    """
    try:
        # Open the file and load the JSON data into a Python object
        with open(file_path, 'r') as file:
            json_data = json.load(file)
        return json_data
    except FileNotFoundError:
        # Raise an error if the file path does not exist
        raise FileNotFoundError(f"File not found: {file_path}")
    except ValueError:
        # Raise an error if the file does not contain valid JSON data
        raise ValueError(f"Invalid JSON data in file: {file_path}")
        
def read_kaggle_csv_by_part(num_fold=5, TRAIN_FILE=None, SIGN_TO_IDX=None):
    kaggle_df = pd.read_csv(TRAIN_FILE)
    kaggle_df.loc[:, 'label'] = kaggle_df.sign.map(SIGN_TO_IDX)
    kaggle_df.loc[:, 'fold' ] = -1

    sgkf = StratifiedGroupKFold(n_splits=num_fold, random_state=42, shuffle=True)
    for i, (train_index, valid_index) in enumerate(sgkf.split(kaggle_df.path, kaggle_df.label, kaggle_df.participant_id)):
        kaggle_df.loc[valid_index,'fold'] = i

    return kaggle_df

def read_christ_csv_by_part(PR_TRAIN_FILE=None, TRAIN_FILE=None):
    christ_df = pd.read_csv(PR_TRAIN_FILE)
    kaggle_df = pd.read_csv(TRAIN_FILE)

    christ_df = christ_df.merge(kaggle_df[['path']], on='path',validate='1:1')
    return christ_df

## Dataset
ROWS_PER_FRAME = 543
def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


def null_collate(batch):
    d = {}
    key = batch[0].keys()
    for k in key:
        d[k] = [b[k] for b in batch]
    d['label'] = torch.LongTensor(d['label'])
    return d