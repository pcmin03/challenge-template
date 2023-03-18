from pathlib import Path
import numpy as np 
import pandas as pd
import torch.nn as nn
import torch
import json

class CFG:
    BASE_PATH = Path('/opt/rsna/data/sign_data/asl-signs') # base path 
    LANDMARK_FILES_DIR = BASE_PATH/ 'train_landmark_files' # land mark path
    TRAIN_FILE = BASE_PATH / 'train.csv' 
    JSON_FILE = BASE_PATH/'sign_to_prediction_index_map.json'
    DROP_Z = True
    NUM_FRAMES = 15
    SEGMENTS = 3
    LEFT_HAND_OFFSET = 468
    POSE_OFFSET = LEFT_HAND_OFFSET+21
    RIGHT_HAND_OFFSET = POSE_OFFSET+33
    ROWS_PER_FRAME = 543

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
        
def get_sign_df(pq_path, invert_y=True):
    sign_df = pd.read_parquet(pq_path)
    
    # y value is inverted (Thanks @danielpeshkov)
    if invert_y: sign_df["y"] *= -1 
        
    return sign_df

def load_relevant_data_subset(pq_path):
    """ Extract x, y, z coordination parquet dataframe

    Args: 
        pq_path (str):
            - The path to read dataframe using pandas
    Returns: 
        np.ndarra: coorindation 
    
    """
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / CFG.ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, CFG.ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


def convert_row(row):
    x = load_relevant_data_subset(row[1].path)
    x = feature_converter(torch.tensor(x)).cpu().numpy()
    return x, row[1].label


class FeatureGen(nn.Module):
    def __init__(self):
        super(FeatureGen, self).__init__()
        pass
    
    def forward(self, x):
        
        face_x = x[:,:468,:].contiguous().view(-1, 468*3)
        lefth_x = x[:,468:489,:].contiguous().view(-1, 21*3)
        pose_x = x[:,489:522,:].contiguous().view(-1, 33*3)
        righth_x = x[:,522:,:].contiguous().view(-1, 21*3)
        
        lefth_x = lefth_x[~torch.any(torch.isnan(lefth_x), dim=1),:]
        righth_x = righth_x[~torch.any(torch.isnan(righth_x), dim=1),:]
        
        x1m = torch.mean(face_x, 0)
        x2m = torch.mean(lefth_x, 0)
        x3m = torch.mean(pose_x, 0)
        x4m = torch.mean(righth_x, 0)

        x1s = torch.std(face_x, 0)
        x2s = torch.std(lefth_x, 0)
        x3s = torch.std(pose_x, 0)
        x4s = torch.std(righth_x, 0)
        
        xfeat = torch.cat([x1m,x2m,x3m,x4m, x1s,x2s,x3s,x4s], axis=0)
        xfeat = torch.where(torch.isnan(xfeat), torch.tensor(0.0, dtype=torch.float32), xfeat)
        
        return xfeat
    
def convert_and_save_data():
    df = pd.read_csv(TRAIN_FILE)
    df['label'] = df['sign'].map(label_map)
    npdata = np.zeros((df.shape[0], 3258))
    nplabels = np.zeros(df.shape[0])
    with mp.Pool() as pool:
        results = pool.imap(convert_row, df.iterrows(), chunksize=250)
        for i, (x,y) in tqdm(enumerate(results), total=df.shape[0]):
            npdata[i,:] = x
            nplabels[i] = y
    
    np.save("feature_data.npy", npdata)
    np.save("feature_labels.npy", nplabels)
    
slicer = [slice(av_set[0],av_set[0]+av_set[1]) for av_set in averaging_sets]
[torch.ones([1,543, 3])[:, av_set[0]:av_set[0]+av_set[1], :].shape for av_set in averaging_sets]
face_slice = slice(0,468)
face_slice = slice(468,CFG.POSE_OFFSET)
face_slice = slice(CFG.POSE_OFFSET,CFG.POSE_OFFSET+33)
face_slice = slice(CFG.POSE_OFFSET,522)


