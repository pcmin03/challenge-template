import pandas as pd
import numpy as np
from src.data.components.preprocess.feature_generator_v2 import PreprocessLayer

from tqdm import tqdm


# Get complete file path to file
def get_file_path(path):
    return f'/opt/sign/challenge-template/data/{path}'

# Source: https://www.kaggle.com/competitions/asl-signs/overview/evaluation
ROWS_PER_FRAME = 543  # number of landmarks per frame

def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

def get_data(file_path, INPUT_SIZE, HAND_IDXS0, LANDMARK_IDXS0,
             N_COLS, N_DIMS):
    # Load Raw Data
    data = load_relevant_data_subset(file_path)
    # Process Data Using Tensorflow
    preprocess_layer = PreprocessLayer(INPUT_SIZE, HAND_IDXS0, 
                                       LANDMARK_IDXS0, N_COLS, N_DIMS)
    data, non_empty_frames_idxs = preprocess_layer(data)
    
    return data, non_empty_frames_idxs


# Get the full dataset
def get_x_y(train_df, N_SAMPLES, INPUT_SIZE, N_COLS, N_DIMS, HAND_IDXS0,LANDMARK_IDXS0):
    # Create arrays to save data
    X = np.zeros([N_SAMPLES, INPUT_SIZE, N_COLS, N_DIMS], dtype=np.float32)
    y = np.zeros([N_SAMPLES], dtype=np.int32)
    NON_EMPTY_FRAME_IDXS = np.full([N_SAMPLES, INPUT_SIZE], -1, dtype=np.float32)

    for row_idx, (file_path, sign_ord) in enumerate(tqdm(train_df[['file_path', 'sign_ord']].values)):
        if row_idx % 5000 == 0:
            print(f'Generated {row_idx}/{N_SAMPLES}')

        data, non_empty_frame_idxs = get_data(file_path, INPUT_SIZE, HAND_IDXS0, LANDMARK_IDXS0,
                                                N_COLS, N_DIMS)
        X[row_idx] = data
        y[row_idx] = sign_ord
        NON_EMPTY_FRAME_IDXS[row_idx] = non_empty_frame_idxs
        if np.isnan(data).sum() > 0:
            print(row_idx)
            return data

    # Save X/y
    np.save('X.npy', X)
    np.save('y.npy', y)
    np.save('NON_EMPTY_FRAME_IDXS.npy', NON_EMPTY_FRAME_IDXS)
    
    return X, y, NON_EMPTY_FRAME_IDXS