"""
Second version of a logistic regression model to predict
the outcome of a League of Legends match.
Characteristics:
- Dataset starts from previous version.
- Highly correlated features removed.
- Normalization applied (z-scores).
- Check for class imbalance.
- No feature engineering.
- Light hyperparameter tuning.
- Check for coefficient significance.
Author: Luis Adrián Uribe Cruz
"""
# Import necessary libraries
import time
import pandas as pd
from LogisticRegression import LogReg


def main():
    """
    Main class for all data loading, preprocessing, model training,
    and evaluation.
    """
    # Loaddata
    lolset = pd.read_csv("Datasets/redundantDataset.csv")

    # Correlation dropping
    lolset = lolset.drop(columns=[
        "blueAssists", "blueTotalGold", "blueTotalExperience",
        "blueExperienceDiff", "redAssists", "redTotalGold",
        "redTotalExperience"
        ])

    # Print class balance
    print("Class balance:")
    print(lolset["blueWins"].value_counts(normalize=True))
    time.sleep(4)

    # Initialize model
    lolmodel = LogReg(lr=0.05, maxItr=10000)
    lolmodel.dataLoader(lolset, "blueWins")
    lolmodel.normalize()  # Z-score normalization

    # Train and evaluate model
    lolmodel.fit()
    lolmodel.evaluate()
    time.sleep(2)

    # Plot evaluation metrics
    lolmodel.convergencePlot("Outputs/decorrelated_Convergence.jpg")
    lolmodel.rocCurvePlot("Outputs/decorrelated_ROC.jpg")
    lolmodel.metricPlot(lolmodel.trainMetrics_,
                        "Outputs/decorrelated__train_Metrics.jpg")
    lolmodel.metricPlot(lolmodel.testMetrics_,
                        "Outputs/decorrelated__test_Metrics.jpg")
    lolmodel.confusionPlot(lolmodel.testConfusion_,
                           "Outputs/decorrelated_Confusion.jpg")
    time.sleep(3)

    # Print feature significance
    print("Model results:")
    cof = lolmodel.significantFeatures()
    print(cof[cof["PValue"] < 0.05])

    # Save model
    lolmodel.saveModel("Models/decorrelatedModel.csv")
    lolset.corr().to_excel("Outputs/decorrelatedCorrelations.xlsx")
    lolset.to_csv("Datasets/decorrelatedDataset.csv", index=False)
    print("Model and data saved.")


if __name__ == "__main__":
    main()
