#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Main class for logistic regression
class LogReg:
  def __init__(self, lr=0.01, tolerance=1e-6, maxItr=1000, threshold=0.5):
    """
    Class constructor.

    Parameters:
    - lf = Learning rate (default: 0.01)
    - tolerance = Tolerance for convergence (default: 1e-6)
    - maxItr = Maximum number of iterations (default: 1000)
    - threshold = Threshold for class prediction (default: 0.5)
    It returns nothing.
    """
    #Hyperparameters
    self.lr = lr
    self.tolerance = tolerance
    self.maxItr = maxItr
    self.threshold = threshold

    #Model parameters
    self.coef_ = None
    self.intercept_ = 0.0

    #Model performance
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
    Prepares a pandas DataFrame into normalized, separated data for training and testing
    Parameters:
    - data = Base DataFrame, assumed to be filtered and cleaned already.
    - targetCol = Column index or name to dependent variable
    - trainSize = Percentage from 0 to 1 of the data to be used for training
    It returns nothing.
    """
    #Error validation
    if not 0 < trainSize + valSize < 1:
      raise ValueError("trainSize and valSize must be between 0 and 1")
    if targetCol not in data.columns:
      raise ValueError("targetCol not in DataFrame columns")

    #Separation indexes
    if trainSize + valSize == 1:
      trainSize -= 0.05 #Avoids having no test set
    idxTrain = int(len(data) * trainSize)
    idxVal = int(len(data) * (trainSize + valSize))

    #Separation into X features and Y target
    Xtemp = data.drop(columns=[targetCol]).values
    ytemp = data[targetCol].values.reshape(-1, 1)

    #Separation of Training, Validation and Test sets
    XTrain, XVal, XTest = Xtemp[:idxTrain], Xtemp[idxTrain:idxVal], Xtemp[idxVal:]
    yTrain, yVal, yTest = ytemp[:idxTrain], ytemp[idxTrain:idxVal], ytemp[idxVal:]

    #Initialize coefficients
    features = XTrain.shape[1]
    np.random.seed(42)
    self.coef_ = np.random.randn(features, 1)

    #Save arrays
    self.XTrain = XTrain
    self.yTrain = yTrain

    self.XVal = XVal
    self.yVal = yVal

    self.XTest = XTest
    self.yTest = yTest


  def loss(self, features, target):
    """
    Calculates the logistic loss with given features and target.

    Parameters:
    - features = Array of features.
    - target = Array of target values.
    It returns the loss value.
    """
    #Error validation
    if self.coef_ is None:
      raise ValueError("Model not initialized yet")

    yPred = self.predict(features)
    return - np.mean(target * np.log(yPred) + (1 - target) * np.log(1 - yPred))
    # - np.mean(self.yTrain * np.log(yPred) + (1 - self.yTrain) * np.log(1 - yPred))


  def predict(self, features):
    """
    Calculates the model logit with a given array of features.

    Parameters:
    - features = Array of features.
    It returns a probability for a class.
    """
    #Error validation
    if self.coef_ is None:
      raise ValueError("Model not initialized yet")

    return self._sigmoid(features.dot(self.coef_) + self.intercept_)


  def fitModel(self):
    """
    Train proccess of the model, running gradient descent until convergence or maximum iterations.

    It receives no parameters.
    It returns nothing.
    """
    #Error validations
    if self.XTrain is None:
      raise ValueError("Data not loaded yet")

    for i in range(self.maxItr):
      #Model prediction
      yPred = self.predict(self.XTrain)

      #Loss calculation
      loss = - np.mean(self.yTrain * np.log(yPred) + (1 - self.yTrain) * np.log(1 - yPred))
      self.losses_.append(loss)

      #Gradient calculation
      coefGrad = self.XTrain.T.dot(yPred - self.yTrain) / self.XTrain.shape[0]
      interGrad = np.mean(yPred - self.yTrain)

      #Update parameters
      self.coef_ -= self.lr * coefGrad
      self.intercept_ -= self.lr * interGrad

      if (i > 0) and (abs(self.losses_[i-1] - loss) < self.tolerance):
        self.iterations_ = i
        break #Stops before if converges
      print(f"Iteration {i+1}: Loss = {loss}.")

    self.iterations_ = self.maxItr
    print(f"Done training after {i+1} iterations.")


  def predictClass(self, features):
    """
    Transforms prediction probability into a class prediction.

    Parameters:
    - features = Array of features.
    It returns a boolean prediction as a number, not a True/False.
    """
    #Error validations
    if self.coef_ is None:
      raise ValueError("Model not trained yet")

    return (self.predict(features) >= self.threshold).astype(int)


  def convergence(self):
    """
    Graphs the model´s convergence.

    It receives no parameters.
    It returns nothing.
    """
    #Error validation
    if self.losses_ is None:
      raise ValueError("Model not trained yet")

    plt.figure(figsize=(10, 6))
    plt.plot(self.losses_, label=f"Tolerance:  {self.tolerance}\nLearning rate: {self.lr}")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Model training convergence")

    plt.legend()
    plt.show()


  def accuracy(self, Xfeatures, Ytarget):
    """
    Calculates the performance of the model with given data.

    Parameters:
    -Xfeatures = array of independent values
    -Ytarget = array of dependent values
    It returns two dictionaries with confusion values and metrics.
    """
    #Error validation
    if self.coef_ is None:
      raise ValueError("Model not trained yet")

    confusion = {"TP":0, "TN":0, "FP":0, "FN":0}
    metrics = {"Accuracy":0, "Precision":0, "Recall":0, "FPR": 0, "F1":0}

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

    TP, TN, FP, FN = confusion["TP"], confusion["TN"], confusion["FP"], confusion["FN"]
    metrics["Accuracy"] = (TP + TN) / (TP + TN + FP + FN)
    metrics["Precision"] = (TP / (TP + FP)) if (TP + FP) != 0 else 0
    metrics["Recall"] = (TP / (TP + FN)) if (TP + FN) != 0 else 0
    metrics["FPR"] = (FP / (FP + TN)) if (FP + TN) != 0 else 0
    metrics["F1"] = (2 * (metrics["Precision"] * metrics["Recall"]) / (metrics["Precision"] + metrics["Recall"])) if (metrics["Precision"] + metrics["Recall"]) != 0 else 0

    return confusion, metrics


  def evaluate(self):
    """
    Evaluates the model´s performance in both train and test sets.
    Receives the True Positives, True Negatives, False Positives and False Negatives for confusion.
    Receives the Accuracy, Precision, Recall and F1 for metrics.

    It receives no parameters.
    It returns nothing.
    """
    #Error validations
    if self.coef_ is None:
      raise ValueError("Model not trained yet")

    self.trainConfusion_, self.trainMetrics_ = self.accuracy(self.XTrain, self.yTrain)

    self.testConfusion_, self.testMetrics_ = self.accuracy(self.XTest, self.yTest)


  def confusionPlot(self, confusion):
    """
    Creates an annotated heatmap of the confusion matrix.

    Parameters:
    -confusion: Confusion matrix dictionary.
    It returns nothing.
    """
    pTags = ["Positive", "Negative"]
    rTags = ["Negative", "Positive"]
    data = np.array([[confusion["FP"], confusion["TN"]],
                    [confusion["TP"], confusion["FN"]]])


    fig, ax = plt.subplots()
    im = ax.imshow(data)

    ax.set_xticks(range(len(pTags)), labels=pTags)
    ax.set_yticks(range(len(rTags)), labels=rTags)

    for i in range(len(rTags)):
      for j in range(len(pTags)):
        text = ax.text(j, i, data[i, j],
                       ha="center", va="center", color="w")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Real")
    ax.set_title("Confusion Matrix")

    plt.show()


  def metricPlot(self, metrics):
    """
    Plots the precision, recall and F1 score.

    Parameters:
    -metrics: Metrics dictionary.
    It returns nothing.
    """
    #Error validations
    if metrics is None:
      raise ValueError("Metrics not calculated yet")

    plt.bar(["Accuracy", "Precision", "Recall", "F1"], [metrics["Accuracy"], metrics["Precision"], metrics["Recall"], metrics["F1"]])
    plt.title("Model performance metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.show()


  def rocCurve(self):
    """
    Graphs the ROC curves and calculates its respective AUC within the Test set.

    It receives no parameters.
    It returns nothing.
    """
    #Error validations
    if self.coef_ is None:
      raise ValueError("Model not trained yet")

    temp = self.threshold
    thresholds = np.linspace(0, 1, 50)
    tprs = []
    fprs = []

    #Iterate the accuracy and metric tests with all different thresholds
    for threshold in thresholds:
      self.threshold = threshold
      _, metrics = self.accuracy(self.XTest, self.yTest)
      tprs.append(metrics["Recall"])
      fprs.append(metrics["FPR"])

    # Creates a sorted copy so that integrations resolves correctly
    sortIdx = np.argsort(fprs)
    auc = np.trapz(np.array(tprs)[sortIdx], np.array(fprs)[sortIdx])


    self.threshold = temp

    plt.figure()
    plt.plot(fprs, tprs, label=f"AUC: {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", label="Randomness")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

  def saveModel(self, filename):
    """
    Exports the model´s coeficients and intercept into a CSV file.

    Parameters:
    -filename: Name of the file to save the model.
    It returns nothing.
    """
    #Error validation
    if ".csv" not in filename:
      filename += ".csv"

    file = open(filename, "w")

    for coef in self.coef_:
      file.write(f"{coef[0]},")
    file.write(str(self.intercept_))

    print(f"Model saved as {filename}")
    file.close()

  def loadModel(self, filename):
    """
    Imports the model´s coeficients and intercept from a CSV file.

    Parameters:
    -filename: Name of the file to load the model.
    It returns nothing.
    """
    #Error validation
    if ".csv" not in filename:
      filename += ".csv"

    file = open(filename, "r")

    coefs = file.readline().split(",")
    temp = np.array(coefs[:-1]).astype(float).reshape(-1, 1)
    self.coef_ = temp
    self.intercept_ = temp[-1]

    print(f"Model loaded from {filename}")
    file.close()