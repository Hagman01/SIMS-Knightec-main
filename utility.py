import pandas as pd
from sklearn.model_selection import train_test_split  

def crossValidation(data_frame, colName, n=10, random=True):
    """
    Splits the given DataFrame object into n-number of X and Y sets to be used for cross-validation

    :param pandas.DataFrame data_frame: The DataFrame object to split
    :param string colName: The name of the target feature
    :param int n: the number of different sets to split the set into, default 10
    :param bool random: If the DataFrame should be randomized or not, deafult True
    :return: n number of list for x and y values
    """
    # Randomize the dataframe
    data_frame = data_frame.sample(frac=1)

    # Split into x and y
    x = data_frame.loc[:, data_frame.columns != colName]
    y = data_frame[colName]

    length = int(y.shape[0]/10)
    x_sets = []
    y_sets = []

    for i in range(n-1):
        x_sets.append(x[i*length:(i+1)*length])
        y_sets.append(y[i*length:(i+1)*length])
    x_sets.append(x[9*length:])
    y_sets.append(y[9*length:])

    return x_sets, y_sets


def MLdata(data):
    #x = data.loc[:, data.columns != colName]
    #y = data[colName]
    x= data.iloc[:, :-1].values  
    y= data.iloc[:, -1].values  
    x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.9, random_state=42)
    return x_train, x_test, y_train, y_test