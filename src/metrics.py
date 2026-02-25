"""
This module contains the implementation of the metrics used to evaluate the performance of the model.

Author: Luis Adrián Uribe Cruz
"""

import numpy as np


def confusionMatrix(yTrue, yPredClass):
    """
    Computes the confusion matrix for binary classification.
    Parameters:
        yTrue: np.array, true target values.
        yPredClass: np.array, predicted target binary values.
    
    Returns:
    - A confusion matrix as a 2x2 dictionary.
    """

    TP = np.sum((yTrue == 1) & (yPredClass == 1))
    TN = np.sum((yTrue == 0) & (yPredClass == 0))
    FP = np.sum((yTrue == 0) & (yPredClass == 1))
    FN = np.sum((yTrue == 1) & (yPredClass == 0))

    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN}


def accuracy(confMatrix):
    """
    Computes the accuracy metric from the confusion matrix.

    Parameters:
        confMatrix: dict, confusion matrix with keys 'TP', 'TN', 'FP', 'FN'.

    Returns:
    - Accuracy as a float.
    """

    return (confMatrix["TP"] + confMatrix["TN"]) / sum(confMatrix.values())


def precision(confMatrix):
    """
    Computes the precision metric from the confusion matrix.

    Parameters:
        confMatrix: dict, confusion matrix with keys 'TP', 'TN', 'FP', 'FN'.

    Returns:
    - Precision as a float.
    """

    if (confMatrix["TP"] + confMatrix["FP"]) == 0:
        return 0

    else:
        return confMatrix["TP"] / (confMatrix["TP"] + confMatrix["FP"])


def recall(confMatrix):
    """
    Computes the recall metric from the confusion matrix.

    Parameters:
        confMatrix: dict, confusion matrix with keys 'TP', 'TN', 'FP', 'FN'.

    Returns:
    - Recall as a float.
    """

    if (confMatrix["TP"] + confMatrix["FN"]) == 0:
        return 0
    else:
        return confMatrix["TP"] / (confMatrix["TP"] + confMatrix["FN"])



def fpr(confMatrix):
    """
    Computes the false positive rate (FPR) from the confusion matrix.

    Parameters:
        confMatrix: dict, confusion matrix with keys 'TP', 'TN', 'FP', 'FN'.

    Returns:
    - FPR as a float.
    """
    if (confMatrix["FP"] + confMatrix["TN"]) == 0:
        return 0
    else:
        return confMatrix["FP"] / (confMatrix["FP"] + confMatrix["TN"])


def f1(confMatrix):
    """
    Computes the F1 score from the confusion matrix.
    
    Parameters:
        confMatrix: dict, confusion matrix with keys 'TP', 'TN', 'FP', 'FN'.
    
    Returns:
    - F1 score as a float.
    """
    
    prec = precision(confMatrix)
    rec = recall(confMatrix)
    
    if prec + rec == 0:
        return 0
    else:
        return 2 * (prec * rec) / (prec + rec)


def rocAuc(yTrue, yPredProb):
    """
    Computes the ROC AUC score for binary classification.

    Parameters:
        yTrue: np.array, true target values.
        yPredProb: np.array, predicted probabilities for the positive class.
    Returns:
    - AUC score as a float.
    - tuple of (FPR, TPR) for ROC curve plotting.
    """
    
    thresholds = np.linspace(0, 1, num=100)
    tprs, fprs = [], []
    
    for threshold in thresholds:
        yPredClass = (yPredProb >= threshold).astype(int)
        confMatrix = confusionMatrix(yTrue, yPredClass)
        tprs.append(recall(confMatrix))
        fprs.append(fpr(confMatrix))
    
    # Sort FPR and TPR for AUC calculation
    sortIdx = np.argsort(fprs)
    auc = np.trapz(np.array(tprs)[sortIdx], np.array(fprs)[sortIdx])
    
    return auc, (fprs, tprs)