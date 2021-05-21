import argparse
import glob
import os
import random

import numpy as np

from utils import get_module_logger
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import shutil
def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /mnt/data
    """
    # TODO: Implement function
    #Get data from filename
    try:
        files = [filename for filename in glob.glob(f'{data_dir}/*.tfrecord')]
    except Exception as err:
        print("Unable to access file")
    np.random.shuffle(files)
    
    # spliting files
    train_files, val_file, test_file = np.split(files, [int(.75*len(files)), int(.9*len(files))])
    
    # create dirs and move data files into them
    train = os.path.join(data_dir, 'train')
    # check if dirs exist and then move the data files in dirs
    try:
        if os.path.exists(train):
            os.makedirs(train)
    except:
        os.makedirs(train,exist_ok=True)
    
    for file in train_files:
        shutil.move(file, train)
    
    val = os.path.join(data_dir, 'val')
    
    try:
        if os.path.exists(val):
            os.makedirs(val)
    except:
        os.makedirs(val,exist_ok=True)
    
    for file in val_file:
        shutil.move(file, val)
    
    test = os.path.join(data_dir, 'test')
    os.makedirs(test, exist_ok=True)
    for file in test_file:
        shutil.move(file, test) 

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)