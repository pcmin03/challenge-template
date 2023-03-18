from pathlib import Path
import numpy as np 
import pandas as pd
import torch.nn as nn
import torch
import json
from src.utils.asl_utils import * 
from src.data.components.preprocess.featuer_generator import TFFeatureGen, TorchFeatureGen 
try: 
    import tensorflow as tf
except: 
    assert "Need install tensorflow"

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

def get_sign_df(pq_path, invert_y=True):
    sign_df = pd.read_parquet(pq_path)
    
    # y value is inverted (Thanks @danielpeshkov)
    if invert_y: sign_df["y"] *= -1 
        
    return sign_df

def load_relevant_data_subset(pq_path, rows_per_frame):
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

def convert_row(row, feature_converter):
    x = load_relevant_data_subset(row[1].path)
    x = feature_converter(torch.tensor(x)).cpu().numpy()
    return x, row[1].label

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
