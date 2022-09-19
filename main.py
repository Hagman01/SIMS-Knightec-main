from distutils.command.clean import clean
from dataPreprocessing import *
from ML import *
from utility import *
import pandas as pd 
import numpy as np

def main():
    # Read data
    filename = './data-cleaned/optiRun_c11.csv'
    df = pd.read_csv(filename)

    # Encode with onehotencoder
    data, env = onehotencoder(df)
    print(data)
    # Get training data
    col = 'Closed Reason'
    value = MLdata(data)

    # Methods
    GradientBoosting(value[0], value[1], value[2], value[3])
    GNB(value[0], value[1], value[2], value[3])
    Multinomial(value[0], value[1], value[2], value[3])
    HistGradientBoosting(value[0], value[1], value[2], value[3])
    decisiontree(value[0], value[1], value[2], value[3])

if __name__ == '__main__':
    main()