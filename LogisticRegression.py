"""
Logistic Regression implementation from scratch in Python.
Author: Luis Adrián Uribe Cruz
"""

# Imports
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# Main class for logistic regression


class LogReg:
    def __init__(self, lr=0.01, tolerance=1e-6, maxItr=1000, threshold=0.5):
        """
        Class constructor.

        Parameters:
        - lf = Learning rate (default: 0.01)
        - tolerance = Tolerance for convergence (default: 1e-6)
        - maxItr = Maximum number of iterations (default: 1000)
        - threshold = Threshold for class prediction (default: 0.5)
        It returns nothing, but prepares variables for training and testing.
        """
        # Hyperparameters
        self.lr = lr
        self.tolerance = tolerance
        self.maxItr = maxItr
        self.threshold = threshold

        # Model parameters
        self.coef_ = None
        self.intercept_ = 0.0

        # Data normalization
        self.norm_ = None
        self.method = []
        self.std = None
        self.mean = None
        self.max = None
        self.min = None

        # Model performance
        self.losses_ = []
        self.valLosses_ = []

        self.trainConfusion_ = None
        self.trainMetrics_ = None

        self.valConfusion_ = None
        self.valMetrics_ = None

        self.testConfusion_ = None
        self.testMetrics_ = None


    def _sigmoid(self, z):
        """
        Function calculation the sigmoid of an expression.
        Parameters:
        - z = Expression.
        It returns the sigmoid of the expression.
        """
        return 1 / (1 + np.exp(-z))

    def dataLoader(self, data, targetCol, trainSize=0.6, valSize=0.2):
        """
        Prepares a pandas DataFrame into normalized,
        separated data for training and testing

        Parameters:
        - data = Base DataFrame, assumed to be filtered and cleaned already.
        - targetCol = Column index or name to dependent variable
        - trainSize = Percentage from 0 to 1 of the
                      train section (default: 0.6)
        - valSize = Percentage from 0 to 1 of the
                      validation section (default: 0.2)
        It returns nothing, but saves the divided datasets.
        """
        # Error validation
        if not 0 < trainSize + valSize <= 1:
            raise ValueError("Provided sizes are not valid.")

        # Separation indexes
        if trainSize + valSize == 1:
            trainSize -= 0.05  # Avoids having no test set
        idxTrain = int(len(data) * trainSize)
        idxVal = int(len(data) * (trainSize + valSize))

        # Separation into X features and Y target
        Xtemp = data.drop(columns=[targetCol]).values
        ytemp = data[targetCol].values.reshape(-1, 1)

        # Save feature names
        self.features_ = data.drop(columns=[targetCol]).columns
        self.features_ = self.features_.insert(0, "Intercept")

        # Separation of Training, Validation and Test sets
        XTrain, XVal = Xtemp[:idxTrain], Xtemp[idxTrain:idxVal]
        XTest = Xtemp[idxVal:]
        yTrain, yVal = ytemp[:idxTrain], ytemp[idxTrain:idxVal]
        yTest = ytemp[idxVal:]

        # Initialize coefficients
        features = XTrain.shape[1]
        np.random.seed(42)
        self.coef_ = np.random.randn(features, 1)

        # Save arrays
        self.XTrain = XTrain.astype(float)
        self.yTrain = yTrain

        self.XVal = XVal.astype(float)
        self.yVal = yVal

        self.XTest = XTest.astype(float)
        self.yTest = yTest

        # Normalization parameters
        self.mean = XTrain.mean(axis=0)
        self.std = XTrain.std(axis=0)
        self.min = XTrain.min(axis=0)
        self.max = XTrain.max(axis=0)
        self.norm_ = False

    def normalize(self, method="z-score", force=False):
        """
        Normalizes the data using either z-score or min-max normalization.

        Parameters:
        - method = "z-score" or "min-max" (default: "z-score")
        It returns nothing, but normalizes the data in place.
        """
        # Error validation
        if self.XTrain is None:
            raise ValueError("Data not loaded yet")
        if method not in ["z-score", "min-max"]:
            raise ValueError("Unrecognized normalization method")
        if self.norm_ and not force:
            warnings.warn(
                "Data already normalized, use force=True to overwrite",
                UserWarning
            )

        if method == "z-score":  # Avoid division by zero
            self.XTrain = np.divide(
                (self.XTrain - self.mean),
                self.std,
                out=np.zeros_like(self.XTrain),
                where=self.std != 0,
            )
            self.XVal = np.divide(
                (self.XVal - self.mean),
                self.std,
                out=np.zeros_like(self.XVal),
                where=self.std != 0,
            )
            self.XTest = np.divide(
                (self.XTest - self.mean),
                self.std,
                out=np.zeros_like(self.XTest),
                where=self.std != 0,
            )

        else:  # Avoid division by zero in "min-max"
            self.XTrain = np.divide(
                (self.XTrain - self.min),
                (self.max - self.min),
                out=np.zeros_like(self.XTrain),
                where=(self.max - self.min) != 0,
            )
            self.XVal = np.divide(
                (self.XVal - self.min),
                (self.max - self.min),
                out=np.zeros_like(self.XVal),
                where=(self.max - self.min) != 0,
            )
            self.XTest = np.divide(
                (self.XTest - self.min),
                (self.max - self.min),
                out=np.zeros_like(self.XTest),
                where=(self.max - self.min) != 0,
            )

        self.norm_ = True
        self.method.append(method)

    def loss(self, features, target, predictions):
        """
        Calculates the logistic loss with given features and target.

        Parameters:
        - features = Array of features.
        - target = Array of target values.
        - predictions = Array of predicted values.
        It returns the loss value.
        """
        # Error validation
        if self.coef_ is None:
            raise ValueError("Model not initialized yet")

        return -np.mean(
            target * np.log(predictions) + (1 - target)
            * np.log(1 - predictions)
        )

    def predict(self, features):
        """
        Calculates the model logit with a given array of features.

        Parameters:
        - features = Array of features.
        It returns a probability for a class.
        """
        # Error validation
        if self.coef_ is None:
            raise ValueError("Model not initialized yet")

        return self._sigmoid(features.dot(self.coef_) + self.intercept_)

    def fitModel(self):
        """
        Train process of the model, running gradient descent until convergence
        or maximum iterations.

        It receives no parameters.
        It returns nothing, but modifies the model coefficients.
        """
        # Error validations
        if self.XTrain is None:
            raise ValueError("Data not loaded yet")

        # Early stopping variables
        patience = 10
        bestLoss = float("inf")
        patienceCounter = 0
        bestCoeffs = None
        bestIntercept = None
        bestIteration = 0

        for i in range(self.maxItr):
            # Model prediction
            yPred = self.predict(self.XTrain)

            # Loss calculation
            lossTrain = self.loss(self.XTrain, self.yTrain, yPred)
            self.losses_.append(lossTrain)

            # Gradient calculation
            coefGrad = self.XTrain.T.dot(yPred - self.yTrain)
            coefGrad /= self.XTrain.shape[0]
            interGrad = np.mean(yPred - self.yTrain)

            # Update parameters
            self.coef_ -= self.lr * coefGrad
            self.intercept_ -= self.lr * interGrad

            # Early stopping check
            self.valLosses_.append(
                self.loss(self.XVal, self.yVal, self.predict(self.XVal))
            )
            if self.valLosses_[-1] < bestLoss:
                bestLoss = self.valLosses_[-1]
                bestCoeffs = self.coef_.copy()
                bestIntercept = self.intercept_
                bestIteration = i + 1
                patienceCounter = 0
            else:
                patienceCounter += 1
                if patienceCounter >= patience:
                    self.coef_ = bestCoeffs
                    self.intercept_ = bestIntercept
                    self.iterations_ = bestIteration
                    print(f"Early stopping at iteration {i+1}.")
                    print("Restoring best model...")
                    print(f"Iteration {self.iterations_} | Loss {bestLoss}.")
                    break

            # Convergence check
            if ((i > 0) and (abs(self.losses_[i - 1] - lossTrain)
                             < self.tolerance)):
                self.iterations_ = i + 1
                print(f"Done training after {i+1} iterations.")
                break
            print(f"Iteration {i+1}: Loss = {lossTrain}.")

        else:
            self.iterations_ = self.maxItr
            print("Model failed to converge within the maximum iterations.")

    def predictClass(self, features):
        """
        Transforms prediction probability into a class prediction.

        Parameters:
        - features = Array of features.
        It returns a boolean prediction as a number.
        """
        # Error validation
        if self.coef_ is None:
            raise ValueError("Model not trained yet")

        return (self.predict(features) >= self.threshold).astype(int)

    def accuracy(self, Xfeatures, Ytarget):
        """
        Calculates the performance of the model with given data.

        Parameters:
        -Xfeatures = array of independent values
        -Ytarget = array of dependent values
        It returns two dictionaries with confusion values and metrics.
        """
        # Error validation
        if self.coef_ is None:
            raise ValueError("Model not trained yet")

        confusion = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        metrics = {"Accuracy": 0, "Precision": 0, "Recall": 0,
                   "FPR": 0, "F1": 0}

        for row, value in zip(Xfeatures, Ytarget):
            pred = self.predictClass(row)
            if pred == value == 1:
                confusion["TP"] += 1
            elif pred == value == 0:
                confusion["TN"] += 1
            elif pred == 1:
                confusion["FP"] += 1
            else:
                confusion["FN"] += 1

        TP, TN = confusion["TP"], confusion["TN"]
        FP, FN = confusion["FP"], confusion["FN"]
        metrics["Accuracy"] = (TP + TN) / (TP + TN + FP + FN)
        metrics["Precision"] = np.divide(
            TP, (TP + FP), out=np.zeros_like(TP, dtype=float),
            where=(TP + FP) != 0
        )
        metrics["Recall"] = np.divide(
            TP, (TP + FN), out=np.zeros_like(TP, dtype=float),
            where=(TP + FN) != 0
        )
        metrics["FPR"] = np.divide(
            FP, (FP + TN), out=np.zeros_like(TP, dtype=float),
            where=(FP + TN) != 0
        )
        metrics["F1"] = np.divide(
            2 * TP,
            (2 * TP + FP + FN),
            out=np.zeros_like(TP, dtype=float),
            where=(2 * TP + FP + FN) != 0,
        )

        return confusion, metrics

    def evaluate(self):
        """
        Evaluates the model´s performance in both train and test sets.
        Receives the True Positives, True Negatives,
        False Positives and False Negatives for confusion.
        Receives the Accuracy, Precision, Recall and F1 for metrics.

        It receives no parameters.
        It returns nothing, but saves the results in class attributes.
        """
        # Error validations
        if self.coef_ is None:
            raise ValueError("Model not trained yet")

        self.trainConfusion_, self.trainMetrics_ = self.accuracy(
            self.XTrain, self.yTrain
        )

        self.valConfusion_, self.valMetrics_ = self.accuracy(
            self.XVal, self.yVal
        )

        self.testConfusion_, self.testMetrics_ = self.accuracy(
            self.XTest, self.yTest
        )

    def significantFeatures(self):
        """
        Calculates the standard errors of the coefficients to
        determine their significance.
        It receives no parameters.
        It returns a significance dataframe.
        """
        # Error validation
        if self.coef_ is None:
            raise ValueError("Model not trained yet")
        if self.XTrain is None:
            raise ValueError("Data not loaded yet")

        # Copy arrays to add intercept
        trainCopy = np.hstack(
            [np.ones((self.XTrain.shape[0], 1)), self.XTrain]
        )
        coefCopy = np.vstack([np.array([[self.intercept_]]), self.coef_])

        # Calculate standard errors
        pred = self.predict(self.XTrain).flatten()
        V = np.diag((pred * (1 - pred)))
        covMatrix = np.linalg.pinv(
            trainCopy.T @ V @ trainCopy
            )  # Pseudo-inverse to avoid singularities

        # Standard errors
        stdErrors = np.sqrt(np.diag(covMatrix)).reshape(-1, 1)

        # Z-scores and p-values
        zScores = coefCopy / stdErrors
        pValues = 2 * (1 - norm.cdf(np.abs(zScores)))

        # Confidence intervals
        ciLower = coefCopy - 1.96 * stdErrors
        ciUpper = coefCopy + 1.96 * stdErrors

        # Create significance dataframe
        significance = pd.DataFrame({
            "Coefficient": coefCopy.flatten(),
            "StdError": stdErrors.flatten(),
            "ZScore": zScores.flatten(),
            "PValue": pValues.flatten(),
            "CI_Lower": ciLower.flatten(),
            "CI_Upper": ciUpper.flatten()
        }, index=self.features_)

        return significance

    def convergencePlot(self, filename="Convergence.png"):
        """
        Graphs the model´s convergence.

        It receives no parameters.
        It returns nothing, but creates a plot figure file.
        """
        # Error validation
        if self.losses_ is None or self.valLosses_ is None:
            raise ValueError("Model not trained yet")
        if ".png" not in filename:
            filename += ".png"

        plt.figure()

        # Plot training loss
        plt.plot(
            self.losses_,
            label=f"Training Loss (L: {self.losses_[-1]:.4f})",
        )
        # Plot validation loss
        plt.plot(
            self.valLosses_,
            label=f"Validation Loss (L: {self.valLosses_[-1]:.4f})",
            linestyle="--",
        )
        # Plot convergence point
        plt.scatter(
            self.iterations_-1,
            self.losses_[-1],
            color="red",
            marker="o",
            label=f"Convergence (Itr: {self.iterations_})",
            )

        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title(f"Model convergence | LR: {self.lr} | Tol: {self.tolerance}")
        plt.legend()
        plt.grid(True)

        # Save figure
        plt.savefig(filename)

        # Show figure
        plt.show()

    def confusionPlot(self, confusion, filename="Confusion_Matrix.png"):
        """
        Creates an annotated heatmap of the confusion matrix.

        Parameters:
        -confusion: Confusion matrix dictionary.
        It returns nothing, but creates a plot figure file.
        """
        # Error validations
        if confusion is None:
            raise ValueError("Confusion matrix not calculated yet")
        if ".png" not in filename:
            filename += ".png"

        # Heatmap order
        # | FP | TN |
        # | TP | FN |
        pTags = ["Positive", "Negative"]
        rTags = ["Negative", "Positive"]
        data = np.array(
            [[confusion["FP"], confusion["TN"]],
             [confusion["TP"], confusion["FN"]]]
        )
        fig, ax = plt.subplots()
        im = ax.imshow(data, cmap="copper")

        ax.set_xticks(range(len(pTags)), labels=pTags)
        ax.set_yticks(range(len(rTags)), labels=rTags)

        # Annotate heatmap
        for i in range(len(rTags)):
            for j in range(len(pTags)):
                text = ax.text(j, i, data[i, j],
                               ha="center", va="center", color="w")

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Real")
        ax.set_title("Confusion Matrix")

        # Save figure
        plt.savefig(filename)

        # Show figure
        plt.show()

    def metricPlot(self, metrics, filename="Metrics.png"):
        """
        Plots the precision, recall and F1 score.

        Parameters:
        -metrics: Metrics dictionary.
        It returns nothing, but creates a plot figure file.
        """
        # Error validations
        if metrics is None:
            raise ValueError("Metrics not calculated yet")
        if ".png" not in filename:
            filename += ".png"

        # Plot metrics
        plt.bar(
            ["Accuracy", "Precision", "Recall", "F1"],
            [
                metrics["Accuracy"],
                metrics["Precision"],
                metrics["Recall"],
                metrics["F1"],
            ]
        )
        plt.bar_label(plt.gca().containers[0], fmt="%.2f")
        plt.title("Model performance metrics")
        plt.ylabel("Score")
        plt.ylim(0, 1)

        # Save figure
        plt.savefig("Metrics.png")

        # Show figure
        plt.show()

    def rocCurvePlot(self, filename="ROC_Curve.png"):
        """
        Graphs the ROC curves and calculates its respective
        AUC within the Test set.

        It receives no parameters.
        It returns nothing, but creates a plot figure file.
        """
        # Error validations
        if self.coef_ is None:
            raise ValueError("Model not trained yet")
        if ".png" not in filename:
            filename += ".png"

        temp = self.threshold
        thresholds = np.linspace(0, 1, 50)
        tprs = []
        fprs = []

        # Iterate the accuracy and metric tests with all different thresholds
        for threshold in thresholds:
            self.threshold = threshold
            _, metrics = self.accuracy(self.XTest, self.yTest)
            tprs.append(metrics["Recall"])
            fprs.append(metrics["FPR"])

        # Creates a sorted copy so that integrations resolves correctly
        sortIdx = np.argsort(fprs)
        auc = np.trapz(np.array(tprs)[sortIdx], np.array(fprs)[sortIdx])

        # Restore original threshold
        self.threshold = temp

        # Plot ROC curve and AUC
        plt.plot(fprs, tprs, label=f"AUC: {auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--", label="Randomness")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()

        # Save figure
        plt.savefig(filename)

        # Show figure
        plt.show()

    def saveModel(self, filename):
        """
        Exports the model´s hyperparameters, coefficients, intercept
        and loss histories into a CSV file.

        Parameters:
        -filename: Name of the file to save the model.
        It returns nothing, but creates a CSV file.
        """
        # Error validation
        if ".csv" not in filename:
            filename += ".csv"
        file = open(filename, "w")
        if file is None:
            raise ValueError("file not found")

        # Save hyperparameters
        file.write(
            f"{self.lr},{self.tolerance},{self.maxItr},{self.threshold}\n"
            )

        # Save coefficients and intercept
        file.write(",".join(map(str, self.coef_.flatten().tolist())) + ","
                   + str(self.intercept_) + "\n")

        # Save losses
        file.write(",".join(map(str, self.losses_)) + "\n")
        file.write(",".join(map(str, self.valLosses_)) + "\n")

        # Save other parameters
        file.write(f"{self.iterations_},{self.norm_},")
        file.write("".join([m + "," for m in self.method]) + "\n")

        # Save normalization params if applicable
        if self.norm_:
            file.write(",".join(map(str, self.std.tolist())) + "\n")
            file.write(",".join(map(str, self.mean.tolist())) + "\n")
            file.write(",".join(map(str, self.min.tolist())) + "\n")
            file.write(",".join(map(str, self.max.tolist())) + "\n")

        print(f"Model saved as {filename}")
        file.close()

    def loadModel(self, filename):
        """
        Imports the model´s hyperparameters, coefficients, intercept
         and loss history from a CSV file.

        Parameters:
        -filename: Name of the file to load the model.
        It returns nothing, but changes class attributes.
        """
        # Error validation
        if ".csv" not in filename:
            filename += ".csv"
        file = open(filename, "r")
        if file is None:
            raise ValueError("File not found")

        # Load hyperparameters
        params = file.readline().split(",")
        self.lr = float(params[0])
        self.tolerance = float(params[1])
        self.maxItr = int(params[2])
        self.threshold = float(params[3])

        # Load coefficients and intercept
        coefs = file.readline().split(",")
        self.coef_ = np.array(coefs[:-1]).astype(float).reshape(-1, 1)
        self.intercept_ = float(coefs[-1])

        # Load losses
        self.losses_ = np.array(
            file.readline().split(",")
            ).astype(float).tolist()
        self.valLosses_ = np.array(
            file.readline().split(",")
            ).astype(float).tolist()

        # Load other parameters
        params = file.readline().split(",")
        self.iterations_ = int(params[0])
        self.norm_ = params[1] == "True"

        if self.norm_:  # Load normalization params
            self.std = np.array(
                file.readline().split(",")
            ).astype(float)
            self.mean = np.array(
                file.readline().split(",")
            ).astype(float)
            self.min = np.array(
                file.readline().split(",")
            ).astype(float)
            self.max = np.array(
                file.readline().split(",")
            ).astype(float)

            for method in params[2:-1]:  # Normalize keeping order
                self.normalize(method, force=True)

        print(f"Model loaded from {filename}")
        file.close()
