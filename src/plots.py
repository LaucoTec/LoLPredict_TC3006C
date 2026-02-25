"""
    This module contains functions for plotting model performance metrics.
    
    Author: Luis Adrián Uribe Cruz
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def plotLossHistory(lossHistory, filename=None):
    """
    Plots the training and validation loss history.

    Parameters:
    - lossHistory: dict, contains 'train' and 'val' keys with lists of loss values.
    - filename: str, optional, path to save the plot image.
    """
    
    # Error validation
    if not isinstance(lossHistory, dict):
        raise ValueError("lossHistory must be a dictionary with 'train' and 'val' keys.")
    if filename is not None:
        _, ext = os.path.splitext(filename)
        if ext.lower() not in ['.png', '.jpg', '.jpeg', '.pdf']:
            raise ValueError("filename must have a valid image extension (png, jpg, jpeg, pdf).")

    plt.figure(figsize=(10, 6))
    # Plot training and validation loss
    plt.plot(lossHistory['train'], label='Training Loss', color='blue')
    if 'val' in lossHistory:
        plt.plot(lossHistory['val'], label='Validation Loss', color='orange')
        # Plot convergence point
    minLoss = min(lossHistory['val']) if 'val' in lossHistory else min(lossHistory['train'])
    minEpoch = lossHistory['val'].index(minLoss) if 'val' in lossHistory else lossHistory['train'].index(minLoss)
    plt.scatter(minEpoch, minLoss, color='red', label='Convergence', zorder=5)
    plt.title('Loss History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    
    # Save the plot if filename is provided
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)

    plt.show()
    

def plotROCCurve(fprs, tprs, auc=None, filename=None):
    """
    Plots the ROC curve.

    Parameters:
    - fprs: list of false positive rates.
    - tprs: list of true positive rates.
    - auc: float, optional, area under the ROC curve to display in the title.
    - filename: str, optional, path to save the plot image.
    """
    
    # Error validation
    if not (isinstance(fprs, (list, np.ndarray)) and isinstance(tprs, (list, np.ndarray))):
        raise ValueError("fprs and tprs must be lists or numpy arrays.")
    if filename is not None:
        _, ext = os.path.splitext(filename)
        if ext.lower() not in ['.png', '.jpg', '.jpeg', '.pdf']:
            raise ValueError("filename must have a valid image extension (png, jpg, jpeg, pdf).")
    if auc is None:
        label = 'ROC Curve'
    else:
        label = f'ROC Curve (AUC = {auc:.2f})'

    plt.figure(figsize=(8, 8))
    plt.plot(fprs, tprs, label=label, color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    
    # Save the plot if filename is provided
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)

    plt.show()
    

def confusionMatrixHeatmap(confMatrix, filename=None):
    """
    Plots a heatmap of the confusion matrix.

    Parameters:
    - confMatrix: dict, confusion matrix with keys 'TP', 'TN', 'FP', 'FN'.
    - filename: str, optional, path to save the plot image.
    """
    
    # Error validation
    if not isinstance(confMatrix, dict) or not all(k in confMatrix for k in ['TP', 'TN', 'FP', 'FN']):
        raise ValueError("confMatrix must be a dictionary with keys 'TP', 'TN', 'FP', 'FN'.")
    if filename is not None:
        _, ext = os.path.splitext(filename)
        if ext.lower() not in ['.png', '.jpg', '.jpeg', '.pdf']:
            raise ValueError("filename must have a valid image extension (png, jpg, jpeg, pdf).")

    matrix = np.array([[confMatrix['TN'], confMatrix['FP']], [confMatrix['FN'], confMatrix['TP']]])
    
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap='Blues')
    plt.title('Confusion Matrix Heatmap')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks([0, 1], ['Negative (0)', 'Positive (1)'])
    plt.yticks([0, 1], ['Negative (0)', 'Positive (1)'])
    plt.tight_layout()
    
    # Annotate the heatmap
    for i in range(2):
        for j in range(2):
            plt.text(j, i, matrix[i, j], ha='center', va='center', color='black')

    # Save the plot if filename is provided
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)

    plt.show()
    
    
def plotMetrics(metrics: dict, filename=None):
    """
    Plots a bar chart of the given metrics.

    Parameters:
    - metrics: dict, contains metric names as keys and their values as values.
    - filename: str, optional, path to save the plot image.
    """
    
    # Error validation
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be a dictionary with metric names as keys and their values as values.")
    if filename is not None:
        _, ext = os.path.splitext(filename)
        if ext.lower() not in ['.png', '.jpg', '.jpeg', '.pdf']:
            raise ValueError("filename must have a valid image extension (png, jpg, jpeg, pdf).")

    plt.figure(figsize=(8, 6))
    plt.bar(metrics.keys(), metrics.values(), color='skyblue')
    plt.title('Model Performance Metrics')
    plt.ylabel('Value')
    plt.ylim(0, 1)  # Assuming metrics are between 0 and 1
    plt.grid(axis='y')
    plt.tight_layout()

    # Save the plot if filename is provided
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)

    plt.show()