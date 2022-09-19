import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

#exclude_columns = ['Comments', 'Account Number', 'ServiceMax Case/WO URL', 'Triage Time', 'Alert to Acknowledge Time', 'Alarm/Alert', 'Severity', 'Last Service', 'Last Maintenance', 'Region']

# read csv file, clean and drop columns, drop all rows with close reason N/A, convert all data to string type
# @return data frame 
def clean_data(filename, exclude_columns=[], no_duplicate=True):
    df = pd.DataFrame(pd.read_csv(filename).astype(str))
    # drop drop duplicates rows if no_duplicate is true
    if no_duplicate: df.drop_duplicates(inplace=True)
    # drop columns if any
    df = df.loc[:, ~df.columns.isin(exclude_columns)]
    # convert all columns to lowercase
    for col in df:
        df[col] = df[col].str.replace('Ä','A')
        df[col] = df[col].str.replace('Å','A')
        df[col] = df[col].str.replace('Ö','O')
        df[col] = df[col].str.lower()
    
    df.drop(df[(df['Closed Reason'] == 'nan')].index, inplace=True)
    return df

# code from chi.py
# Encode none numerical data to numerical data
def label_data(data_frame):
    column = "Closed Reason"
    data_frame[column] = data_frame[column].replace(["No action required"], 1)
    data_frame[column] = data_frame[column].replace(["Customer resolved on own"], 1)
    data_frame[column] = data_frame[column].replace(["Duplicate event"], 1)
    data_frame[column] = data_frame[column].replace(["Service Case Created"], 0)
    for col in data_frame:
        le = preprocessing.LabelEncoder()
        le.fit(data_frame[col])
        data_frame[col] = le.transform(data_frame[col])
    return data_frame

# Normalize data columnwise in range[0,1] using MinMaxScaler()
def normalize_data(data_frame):
    x = data_frame.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    data_frame = pd.DataFrame(x_scaled, columns=data_frame.columns)
    return data_frame

# Representation of categorical variables as binary vectors.
def onehotencoder(data_frame):
    env = OneHotEncoder(handle_unknown='ignore')
    env.fit(data_frame)
    data_frame = pd.DataFrame(env.transform(data_frame).toarray())
    names = env.get_feature_names_out()
    for i, name in enumerate(names):
        names[i] = name[name.find('_')+1:]
    data_frame.columns=names
    return data_frame, env

def onehot_inv(data_frame, env):
    """
    Inverse transforms a dataframe which is one-hot encoded back to normal.

    :param pandas.DataFrame data_frame: The dataframe object to be transformed
    :param sklearn.OneHotEncoder env: OneHotEncoder object fitted to the original DataFrame
    :return: The original DataFrame object
    """
    return pd.DataFrame(env.inverse_transform(data_frame))


