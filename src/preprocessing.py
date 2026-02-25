"""
Data preprocessing utilities for the project.

Author: Luis Adrián Uribe Cruz
"""

import numpy as np
import pandas as pd


def dataLoad(filePath, targetCol, shuffle=False, randomState=None):
    """
    Loads the dataset from a CSV file and separates features and target variable.

    Parameters:
    - filePath: str, path to the CSV file.
    - targetCol: str, name of the target column in the dataset.
    - shuffle: bool, whether to shuffle the dataset. Defaults to False.
    - randomState: int, random seed for shuffling. Defaults to None.

    Returns:
    - X: pd.DataFrame, features.
    - y: pd.Series, target variable.
    """

    try:
        data = pd.read_csv(filePath)
        if targetCol not in data.columns:
            raise ValueError(f"Target column '{targetCol}' not found in the dataset.")

        if shuffle:
            data = data.sample(frac=1, random_state=randomState).reset_index(drop=True)

        X = data.drop(columns=[targetCol])
        y = data[targetCol]

        return X, y

    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")


def dataSplit(X, y, trainSize=0.6, valSize=0.2):
    """
    Splits the dataset into training, validation, and test sets.
    Test set is the remaining data after allocating training and validation sets.

    Parameters:
    - X: pd.DataFrame, features.
    - y: pd.Series, target variable.
    - trainSize: float, proportion of data to use for training. Defaults to 0.6.
    - valSize: float, proportion of data to use for validation. Defaults to 0.2.

    Returns:
    - XTrain, XVal, XTest: pd.DataFrame, training, validation, and test features.
    - yTrain, yVal, yTest: pd.Series, training, validation, and test target variables.
    """

    if trainSize + valSize >= 1.0:
        raise ValueError("The sum of trainSize and valSize must be less than 1.0.")

    totalSize = len(X)
    trainSize = int(totalSize * trainSize)
    valSize = int(totalSize * valSize)

    XTrain = X.iloc[:trainSize]
    yTrain = y.iloc[:trainSize]

    XVal = X.iloc[trainSize : trainSize + valSize]
    yVal = y.iloc[trainSize : trainSize + valSize]

    XTest = X.iloc[trainSize + valSize :]
    yTest = y.iloc[trainSize + valSize :]

    return XTrain, yTrain, XVal, yVal, XTest, yTest


def normMinMax(XTrain, XVal, XTest):
    """
    Normalizes the features using Min-Max scaling through Train set.

    Parameters:
    - XTrain: pd.DataFrame, training features.
    - XVal: pd.DataFrame, validation features.
    - XTest: pd.DataFrame, test features.

    Returns:
    - XTrainNorm, XValNorm, XTestNorm: pd.DataFrame, normalized features.
    """

    XMin = XTrain.min()
    XMax = XTrain.max()

    XTrainNorm = (XTrain - XMin) / (XMax - XMin)
    XValNorm = (XVal - XMin) / (XMax - XMin)
    XTestNorm = (XTest - XMin) / (XMax - XMin)

    return XTrainNorm, XValNorm, XTestNorm, XMin, XMax


def normZScores(XTrain, XVal, XTest):
    """
    Normalizes the features using Z-Score scaling through Train set.

    Parameters:
    - XTrain: pd.DataFrame, training features.
    - XVal: pd.DataFrame, validation features.
    - XTest: pd.DataFrame, test features.

    Returns:
    - XTrainNorm, XValNorm, XTestNorm: pd.DataFrame, normalized features.
    """

    XMean = XTrain.mean()
    XStd = XTrain.std()

    XTrainNorm = (XTrain - XMean) / XStd
    XValNorm = (XVal - XMean) / XStd
    XTestNorm = (XTest - XMean) / XStd

    return XTrainNorm, XValNorm, XTestNorm, XMean, XStd
