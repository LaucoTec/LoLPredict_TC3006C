"""
Second version of a logistic regression model to predict
the outcome of a League of Legends match.
Characteristics:
- Dataset starts from redundant version.
- Highly correlated features removed.
- Feature engineering applied.
- Normalization applied (z-scores).
- Check for class imbalance.
- Hyperparameter tuning.
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

    # Feature creation
    lolset["blueKDA"] = (lolset["blueKills"] + lolset["blueAssists"]) / lolset[
        "blueDeaths"].replace(0, 1)
    lolset["redKDA"] = (lolset["blueDeaths"] + lolset["redAssists"]) / lolset[
        "blueKills"].replace(0, 1)

    blueVisionControl = lolset["blueWardsPlaced"] / lolset["redWardsDestroyed"].replace(0, 1)
    redVisionControl = lolset["redWardsPlaced"] / lolset["blueWardsDestroyed"].replace(0, 1)
    lolset["visionControlDiff"] = blueVisionControl - redVisionControl

    blueTotalCS = lolset["blueTotalMinionsKilled"] + lolset["blueTotalJungleMinionsKilled"]
    redTotalCS = lolset["redTotalMinionsKilled"] + lolset["redTotalJungleMinionsKilled"]
    lolset["CSDiff"] = blueTotalCS - redTotalCS

    # Drop used features
    lolset = lolset.drop(columns=[
        "blueWardsPlaced", "redWardsPlaced", "blueWardsDestroyed",
        "redWardsDestroyed", "blueTotalMinionsKilled",
        "blueTotalJungleMinionsKilled", "redTotalMinionsKilled",
        "redTotalJungleMinionsKilled"
        ])

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
    lolmodel = LogReg(lr=0.01, maxItr=10000, threshold=0.5)
    lolmodel.dataLoader(lolset, "blueWins")
    lolmodel.normalize()  # Z-score normalization

    # Train and evaluate model
    lolmodel.fit()
    lolmodel.evaluate()
    time.sleep(2)

    # Plot evaluation metrics
    lolmodel.convergencePlot("Outputs/polished_Convergence.jpg")
    lolmodel.rocCurvePlot("Outputs/polished_ROC.jpg")
    # Plot metrics for different thresholds
    # Neutral threshold
    lolmodel.metricPlot(lolmodel.testMetrics_,
                        "Outputs/polished__neutral_Metrics.jpg")
    lolmodel.confusionPlot(lolmodel.testConfusion_,
                           "Outputs/polished_neutral_Confusion.jpg")
    # Strict threshold
    lolmodel.threshold = 0.6
    lolmodel.evaluate()
    lolmodel.metricPlot(lolmodel.testMetrics_,
                        "Outputs/polished__strict_Metrics.jpg")
    lolmodel.confusionPlot(lolmodel.testConfusion_,
                           "Outputs/polished_strict_Confusion.jpg")
    # Lenient threshold
    lolmodel.threshold = 0.4
    lolmodel.evaluate()
    lolmodel.metricPlot(lolmodel.testMetrics_,
                        "Outputs/polished__lenient_Metrics.jpg")
    lolmodel.confusionPlot(lolmodel.testConfusion_,
                           "Outputs/polished_lenient_Confusion.jpg")
    time.sleep(3)

    # Print feature significance
    print("Model results:")
    cof = lolmodel.significantFeatures()
    print(cof[cof["PValue"] < 0.05])

    # Save model
    lolmodel.saveModel("Models/polishedModel.csv")
    lolset.corr().to_excel("Outputs/polishedCorrelations.xlsx")
    lolset.to_csv("Datasets/polishedDataset.csv", index=False)
    print("Model and data saved.")


if __name__ == "__main__":
    main()
