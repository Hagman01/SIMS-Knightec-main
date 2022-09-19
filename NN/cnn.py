import pandas as pd
import tensorflow as tf

# Import functions from dataPreprocessing.py
import sys
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from dataPreprocessing import onehotencoder
from utility import crossValidation

debug = False #Used to print debug messages, default False

def main():
    #Load dataset, dataset has already been feature selected
    df = pd.read_csv("data_cleaned/optiRun_c11.csv")

    if debug:
        print(df.head())
        print('Tensorflow version:', tf.__version__)
        print('GPUs:', tf.config.list_physical_devices('GPU'))
    
    #Transform dataset using OneHotEncoding
    df, env = onehotencoder(df)

    #Split into x and y sets for 10x cross-validation
    x_sets, y_sets = crossValidation(data_frame=df, colName='no action required')
    

if __name__ == '__main__':
    main()