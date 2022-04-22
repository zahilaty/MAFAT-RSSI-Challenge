import numpy as np   
import pandas as pd
from scipy.stats import skew
from scipy.signal import find_peaks

def extract_features(X, bases=None):
    """
    Extract features from data.
    :param X: Dataset of time windows.
    :param bases: Dictionary with values of bool lists of size 7 and keys of the names of the vectors to extract
    features from
    :return: new dataset with extracted features, training feature name list
    """
    # Restructure dataframe to fit preprocessing features extraction functions - changing the dataframe to have one row.
    # Each column is compressed to a list.
    data = pd.DataFrame(columns=X.columns)
    for col in data.columns:
        data.loc[0,col]= np.array(X[col])

    if bases is None:
        bases = {
        'RSSI': [True, False, False, False, True, True, True],
        'RSSI_diffs': [True, True, True, False, True, True, True],
        'RSSI_diffs_abs': [False, False, True, True, True, False, True],
        'RSSI_median_dist': [True, False, True, True, True, False, True]
    } 

    features_name = []
    data['RSSI_diffs'] = data.RSSI.apply(lambda x: x[1:] - x[:-1])
    data['RSSI_diffs_abs'] = data.RSSI.apply(lambda x: abs(x[1:] - x[:-1]))
    data['RSSI_median_dist'] = data.RSSI.apply(lambda x: abs(x - np.median(x)))

    data, features_name = extract_list_feats('RSSI', data, features_name, base=bases['RSSI'])
    data, features_name = extract_list_feats('RSSI_diffs', data, features_name, base=bases['RSSI_diffs'])
    data, features_name = extract_list_feats('RSSI_diffs_abs', data, features_name, base=bases['RSSI_diffs_abs'])
    data, features_name = extract_list_feats('RSSI_median_dist', data, features_name, base=bases['RSSI_median_dist'])

    data['max_count_same_value_RSSI'] = data.RSSI.apply(lambda x: np.max(np.unique(x, return_counts=True)[1]))
    features_name += ['max_count_same_value_RSSI']

    data['RSSI_peaks'] = data.RSSI.apply(lambda x: len(find_peaks(x)[0]))
    features_name += ['RSSI_peaks']

    data['RSSI_diffs_peaks'] = data.RSSI_diffs.apply(lambda x: len(find_peaks(x)[0]))
    features_name += ['RSSI_diffs_peaks']

    data['peak_ratio_diffs_RSSI'] = data.apply(
        lambda x: x['RSSI_diffs_peaks'] / x['RSSI_peaks'] if x['RSSI_peaks'] > 0 else 0, axis=1)
    features_name += ['peak_ratio_diffs_RSSI']

    data['RSSI_values_count'] = data.RSSI.apply(lambda x: len(np.unique(x)))
    features_name += ['RSSI_values_count']

    return data, features_name

def extract_list_feats(list_name: str, data, features_name: list, base=None):
    """
    Extract Features from vector.
    :param list_name: Vector to extract features from.
    :param data: Dataset to extract features from.
    :param features_name: Feature list to add new feature names to.
    :param base: Disable the use of features.
    :return: Data with features, updated feature name list.
    """

    if base is None:
        base = DEFAULT_TRUE_LIST

    data[f'max_{list_name}'] = data[list_name].apply(np.max)
    if base[0]:
        features_name += [f'max_{list_name}']

    data[f'min_{list_name}'] = data[list_name].apply(np.min)
    if base[1]:
        features_name += [f'min_{list_name}']

    data[f'mean_{list_name}'] = data[list_name].apply(np.mean)
    if base[2]:
        features_name += [f'mean_{list_name}']

    data[f'median_{list_name}'] = data[list_name].apply(np.median)
    if base[3]:
        features_name += [f'median_{list_name}']

    data[f'std_{list_name}'] = data[list_name].apply(np.std)
    if base[4]:
        features_name += [f'std_{list_name}']

    data[f'skew_{list_name}'] = data[list_name].apply(skew)
    if base[5]:
        features_name += [f'skew_{list_name}']

    data[f'max_sub_min_{list_name}'] = data[list_name].apply(lambda x: np.max(x) - np.min(x))
    if base[6]:
        features_name += [f'max_sub_min_{list_name}']

    return data, features_name
    
def preprocess(X, RSSI_value_selection):
    """
    Calculate the features on the selected RSSI on the test set
    :param X: Dataset to extract features from.
    :param RSSI_value_selection: Which signal values to use- - in our case it is Average.
    :return: Test x dataset with features
    """
    if RSSI_value_selection=="RSSI_Left":
        X["RSSI"] = X.RSSI_Left
    elif RSSI_value_selection=="RSSI_Right":
        X["RSSI"] = X.RSSI_Right
    elif RSSI_value_selection=="Min":
        X["RSSI"] = X[['RSSI_Left','RSSI_Right']].min(axis=1).values
    elif RSSI_value_selection=="Max":
        X["RSSI"] = X[['RSSI_Left','RSSI_Right']].max(axis=1).values
    else: 
        X["RSSI"] = np.ceil(X[['RSSI_Left','RSSI_Right']].mean(axis=1).values).astype('int')

    X, features_name = extract_features(X)
    X.drop('Device_ID', axis=1, inplace=True)
    return X[features_name]