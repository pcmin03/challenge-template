from pathlib import Path
from tqdm import tqdm
import numpy as np 
import pandas as pd
import torch.nn as nn
import torch
import json
from src.utils.asl_utils import read_json_file
from src.data.components.preprocess.featuer_generator import FeatureGen

try: 
    import tensorflow as tf
except: 
    assert "Need install tensorflow"

class CFG:
    BASE_PATH = Path('/opt/sign/data/sign_data/asl-signs') # base path 
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
    LABEL_MAP = read_json_file(JSON_FILE)
    ## average over the entire face, and the entire 'pose'
    # AVERAGE_SET = [[0, 468], [POSE_OFFSET, 33]]
    # LIP_LANDMARKS = [61, 185, 40, 39, 37,  0, 267, 269, 270, 409,
    #                 291,146, 91,181, 84, 17, 314, 405, 321, 375, 
    #                 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 
    #                 95, 88, 178, 87, 14,317, 402, 318, 324, 308]

    # LEFT_HAND_LANDMARKS = list(range(LEFT_HAND_OFFSET, LEFT_HAND_OFFSET+21))
    # RIGHT_HAND_LANDMARKS = list(range(RIGHT_HAND_OFFSET, RIGHT_HAND_OFFSET+21))

    # POINT_LANDMARKS = [
    #     item 
    #     for sublist in [LIP_LANDMARKS, LEFT_HAND_LANDMARKS, RIGHT_HAND_LANDMARKS] 
    #     for item in 1
    # ]

    # LANDMARKS = len(POINT_LANDMARKS) + len(AVERAGE_SET)
    # if DROP_Z:
    #     INPUT_SHAPE = (NUM_FRAMES, LANDMARKS * 2)
    # else:
    #     INPUT_SHAPE = (NUM_FRAMES, LANDMARKS * 3)

    # FLAT_INPUT_SHAPE = (INPUT_SHAPE[0] + 2 * (SEGMENTS + 1)) * INPUT_SHAPE[1]



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
    n_frames = int(len(data) / rows_per_frame)
    data = data.values.reshape(n_frames, rows_per_frame, len(data_columns))
    return data.astype(np.float32)

def convert_row(row, feature_converter, row_per_frame):
    x = load_relevant_data_subset(row[1].path, row_per_frame)
    x = feature_converter(torch.tensor(x)).cpu().numpy()
    return x, row[1].label

# def convert_and_save_data():
#     df = pd.read_csv(CFG.TRAIN_FILE)
#     df['label'] = df['sign'].map(CFG.LABEL_MAP)
#     npdata = np.zeros((df.shape[0], 3258))
#     nplabels = np.zeros(df.shape[0])
#     with mp.Pool() as pool:
#         results = pool.imap(convert_row, df.iterrows(), chunksize=250)
#         for i, (x,y) in tqdm(enumerate(results), total=df.shape[0]):
#             npdata[i,:] = x
#             nplabels[i] = y
    
#     np.save("feature_data.npy", npdata)
#     np.save("feature_labels.npy", nplabels)

# def convert_and_save_data(inptu_shape,segment):
#     df = pd.read_csv(CFG.TRAIN_FILE)
#     df['label'] = df['sign'].map(label_map)
#     total = df.shape[0]
#     npdata = np.zeros((total, INPUT_SHAPE[0]*INPUT_SHAPE[1] + (SEGMENTS+1)*INPUT_SHAPE[1]*2))
#     nplabels = np.zeros(total)
#     with mp.Pool() as pool:
#         results = pool.imap(convert_row, df.iterrows(), chunksize=250)
#         for i, (x,y) in tqdm(enumerate(results), total=df.shape[0]):
#             npdata[i,:] = x
#             nplabels[i] = y
            
#     np.save("feature_data.npy", npdata)
#     np.save("feature_labels.npy", nplabels)