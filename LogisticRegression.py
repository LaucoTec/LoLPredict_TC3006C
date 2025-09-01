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
    self.trainConfussion_ = None
    self.trainMetrics_ = None
    self.testConfussion_ = None
    self.testMetrics_ = None


  def _sigmoid(self, z):
    """
    Function calculation the sigmoid of an expression.
    Parameters:
    - z = Expression.
    It returns the sigmoid of the expression.
    """
    return 1 / (1 + np.exp(-z))


  def _zScores(self, temp, mean=None, std=None):
    """
    Function that calculates the z-score of a given array.

    Parameters:
    - temp = Array to be standardized.
    - mean = Array of means per column. If None, calculates it.
    - std = Array of standard deviations per column. If None, calculates it.
    It returns the standardized array, the mean array and the standard deviation array.
    """
    if mean is None:
      mean = np.mean(temp, axis=0)
    if std is None:
      std = np.std(temp, axis=0) + 1e-10

    return (temp - mean) / std, mean, std


  def dataLoader(self, data, targetCol, trainSize=0.8):
    """
    Prepares a pandas DataFrame into normalized, separated data for training and testing
    Parameters:
    - data = Base DataFrame, assumed to be filtered and cleaned already.
    - targetCol = Column index or name to dependent variable
    - trainSize = Percentage from 0 to 1 of the data to be used for training
    It returns nothing.
    """
    #Error validation
    if not 0 < trainSize < 1:
      raise ValueError("trainSize must be between 0 and 1")

    #Separation index for training and testing
    idx = int(len(data) * trainSize)

    #Separation into X features and Y target
    Xtemp = data.drop(columns=[targetCol]).values
    ytemp = data[targetCol].values.reshape(-1, 1)

    #Separation of Training and Testing
    XTrain, XTest = Xtemp[:idx], Xtemp[idx:]
    yTrain, yTest = ytemp[:idx], ytemp[idx:]

    #Standarization trough Z-Scores
    XTrain, self.mean, self.std = self._zScores(XTrain)
    XTest,_,_ = self._zScores(XTest, self.mean, self.std)

    #Initialize coefficients
    features = XTrain.shape[1]
    np.random.seed(42)
    self.coef_ = np.random.randn(features, 1)

    #Save arrays
    self.XTrain = XTrain
    self.yTrain = yTrain
    self.XTest = XTest
    self.yTest = yTest


  def predict(self, features):
    """
    Calculates the model prediction with a given array of features.

    Parameters:
    - features = Array of features.
    It returns a probability for a class.
    """
    #Error validation
    if self.coef_ is None:
      raise ValueError("Model not trained yet")

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
    Graphs the model´s convergence, pointing where.

    It receives no parameters.
    It returns nothing.
    """
    #Error validation
    if self.losses is None:
      raise ValueError("Model not trained yet")

    plt.figure(figsize=(10, 6))
    plt.plot(self.losses_)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Model training convergence")

    plt.annotate(f"Tolerance:  {self.tolerance}\nLearning rate: {self.lr}",
                 xy=(self.iterations_, self.losses_[-1]), xytext=(self.iterations_*0.6, self.losses_[-1]*1.1))

    plt.show()


  def accuracy(self, Xfeatures, Ytarget):
    """
    Calculates the performance of the model with given data.

    Parameters:
    -Xfeatures = array of independent values
    -Ytarget = array of dependent values
    It returns two dictionaries with confussion values and metrics.
    """
    #Error validation
    if self.coef_ is None:
      raise ValueError("Model not trained yet")

    confussion = {"TP":0, "TN":0, "FP":0, "FN":0}
    metrics = {"Accuracy":0, "Precision":0, "Recall":0, "FPR": 0, "F1":0}

    for row, value in zip(Xfeatures, Ytarget):
      pred = self.predictClass(row)
      if pred == value == 1:
        confussion["TP"] += 1
      elif pred == value == 0:
        confussion["TN"] += 1
      elif pred == 1:
        confussion["FP"] += 1
      else:
        confussion["FN"] += 1

    metrics["Accuracy"] = (confussion["TP"] + confussion["TN"]) / Ytarget.shape[0]
    metrics["Precision"] = confussion["TP"] / (confussion["TP"] + confussion["FP"])
    metrics["Recall"] = confussion["TP"] / (confussion["TP"] + confussion["FN"])
    metrics["FPR"] = confussion["FP"] / (confussion["FP"] + confussion["TN"])
    metrics["F1"] = 2 * (metrics["Precision"] * metrics["Recall"]) / (metrics["Precision"] + metrics["Recall"])

    return confussion, metrics


  def evaluate(self):
    """
    Evaluates the model´s performance in both train and test sets.
    Receives the True Positives, True Negatives, False Positives and False Negatives for confussion.
    Receives the Accuracy, Precision, Recall and F1 for metrics.

    It receives no parameters.
    It returns nothing.
    """
    #Error validations
    if self.coef_ is None:
      raise ValueError("Model not trained yet")

    self.trainConfussion_, self.trainMetrics_ = self.accuracy(self.XTrain, self.yTrain)

    self.testConfussion_, self.testMetrics_ = self.accuracy(self.XTest, self.yTest)


  def confusionPlot(self, confussion):
    """
    Creates an annotated heatmap of the confusion matrix.

    Parameters:
    -confussion: Confussion matrix dictionary.
    It returns nothing.
    """
    tags = ["Positive", "Negative"]
    data = np.array([[confussion["TP"], confussion["FP"]], [confussion["FN"], confussion["TN"]]])

    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap="Blues")

    ax.set_xticks(np.arange(len(tags)), labels=tags)
    ax.set_yticks(np.arange(len(tags)), labels=tags)

    for i in range(len(tags)):
      for j in range(len(tags)):
        text = ax.text(j, i, data[i, j], ha="center", va="center", color="w")

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

    auc = np.trapz(tprs, fprs)

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
      file.write(str(coef)+",")
    file.write(str(self.intercept_))

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
    self.coef_ = np.array(float(coefs[:-1]).reshape(-1, 1))
    self.intercept_ = float(coefs[-1])

    file.close()