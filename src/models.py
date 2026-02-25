"""
Models for LoLPredict_TC3006C project.
Includes implementation of Logistic Regression.

Author: Luis Adrián Uribe Cruz
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


class LogisticRegression:
    """
    Logistic Regression model for binary classification.
    """

    def __init__(self, lr=0.01, maxIter=1000, tol=1e-6):
        """
        Class constructor.

         Parameters:
         - lr: float, learning rate for gradient descent. Defaults to 0.01.
         - maxIter: int, maximum number of iterations for training. Defaults to 1000.
         - tol: float, tolerance for convergence. Defaults to 1e-6.
        """

        # Hyperparameters
        self.lr = lr
        self.maxIter = maxIter
        self.tol = tol
        self.threshold = 0.5  # Default threshold for classification

        # Model parameters
        self.coef_ = None
        self.intercept_ = None
        self.lossHistory_ = {
            "train": [],
            "val": [],
        }  # To store loss history for training and validation

    def sigmoid(self, z):
        """
        Computes the sigmoid function for a given input z.

        Parameter:
        - z: Input value(s) to compute sigmoid for.

        Returns:
        - Sigmoid of z.
        """

        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, XVal=None, yVal=None, patience=10, verbose=False):
        """
        Trains the logistic regression model using gradient descent.

        Parameters:
        - X: np.ndarray, features.
        - y: np.array, target variable.
        - XVal: np.ndarray, validation features.
        - yVal: np.array, validation target variable.
        - patience: int, number of iterations to wait for improvement before early stopping. Defaults to 10.
        - verbose: bool, whether to print training progress. Defaults to False.
        """
        # Initialize parameters
        nSamples, nFeatures = X.shape
        self.coef_ = np.zeros((nFeatures, 1))
        self.intercept_ = 0.0

        # Early stopping parameters
        bestLoss = np.inf
        patienceCounter = 0
        bestCoef = None
        bestIntercept = None
        bestIteration = 0

        for i in range(self.maxIter):
            yPred = self.predProb(X)

            # Compute loss and store history
            loss = self.loss(y, yPred)
            self.lossHistory_["train"].append(loss)

            # Gradient computation
            coefGrad = X.T.dot(yPred - y) / nSamples
            interceptGrad = np.mean(yPred - y)

            # Update parameters
            self.coef_ -= self.lr * coefGrad
            self.intercept_ -= self.lr * interceptGrad
            if verbose:
                print(f"Iteration {i+1}/{self.maxIter}, Loss: {loss:.6f}")

            # Early stopping check (if validation data is provided)
            if XVal is not None and yVal is not None:
                valLoss = self.loss(yVal, self.predProb(XVal))
                self.lossHistory_["val"].append(valLoss)

                if valLoss < bestLoss:
                    bestLoss = valLoss
                    bestCoef = self.coef_.copy()
                    bestIntercept = self.intercept_
                    bestIteration = i
                    patienceCounter = 0
                else:
                    patienceCounter += 1
                    if patienceCounter >= patience:
                        self.coef_ = bestCoef
                        self.intercept_ = bestIntercept
                        if verbose:
                            print(f"Early stopping at iteration {bestIteration}")
                        break

            # Check for convergence (if validation data is not provided)
            else:
                if (
                    i > 0
                    and abs(
                        self.lossHistory_["train"][-2] - self.lossHistory_["train"][-1]
                    )
                    < self.tol
                ):
                    if verbose:
                        print(f"Convergence reached at iteration {i+1}")
                    break

        else:
            if verbose:
                print("Maximum iterations reached without convergence.")

    def predProb(self, X):
        """
        Calculates the model probabilities for the input features X.

        Parameter:
        - X: np.ndarray, features.

        Returns:
        - Probabilities for the input features.
        """

        # Error validation
        if self.coef_ is None:
            raise ValueError("Model not initialized yet")

        return self.sigmoid(X.dot(self.coef_) + self.intercept_)

    def predClass(self, X):
        """
        Predicts class labels for the input features X based on the predicted probabilities.

        Parameter:
        - X: np.ndarray, features.

        Returns:
        - Predicted class labels (0 or 1).
        """

        yPredProb = self.predProb(X)
        return (yPredProb >= self.threshold).astype(int)

    def loss(self, yTrue, yPred):
        """
        Computes the binary cross-entropy loss.

        Parameters:
        - yTrue: pd.Series or np.array, true target values.
        - yPred: np.array, predicted probabilities.

        Returns:
        - Binary cross-entropy loss.
        """

        # Avoid log(0)
        epsilon = 1e-15
        yPred = np.clip(yPred, epsilon, 1 - epsilon)

        return -np.mean(yTrue * np.log(yPred) + (1 - yTrue) * np.log(1 - yPred))

    def significantFeatures(self, X):
        """
        Returns the statistical significance of each feature in the model.

        Parameters:
        - X: pd.DataFrame, features.

        Returns:
        - DataFrame with statistical significance of each feature.
        """

        # Error validation
        if self.coef_ is None:
            raise ValueError("Model not initialized yet")

        # Copy of data to add intercept term
        XCopy = np.hstack([np.ones((X.shape[0], 1)), X.values])
        coefCopy = np.vstack(([[self.intercept_]], self.coef_))

        # Calculate standard error of coefficients
        preds = self.predProb(X).flatten()
        V = np.diag((preds * (1 - preds)))
        covMatrix = np.linalg.pinv(XCopy.T @ V @ XCopy)
        stdErrors = np.sqrt(np.diag(covMatrix)).reshape(-1, 1)

        # Calculate z-scores and p-values
        zScores = np.abs(coefCopy / stdErrors)
        pValues = 2 * (1 - norm.cdf(zScores))

        # Confidence interval check for significance (95% confidence level)
        ciLower = coefCopy - 1.96 * stdErrors
        ciUpper = coefCopy + 1.96 * stdErrors

        # Build significance dataframe
        features = ["Intercept"] + list(X.columns)
        significance = pd.DataFrame(
            {
                "Coefficient": coefCopy.flatten(),
                "StdError": stdErrors.flatten(),
                "ZScore": zScores.flatten(),
                "PValue": pValues.flatten(),
                "CI_Lower": ciLower.flatten(),
                "CI_Upper": ciUpper.flatten(),
            },
            index=features,
        )

        return significance
