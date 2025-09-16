"""
First version of a logistic regression model to predict
the outcome of a League of Legends match.
Characteristics:
- No data transformation beyond dropping directly redundant features.
- Other highly correlated features retained.
- Only normalization applied (z-scores).
- No possible class imbalance handling.
- No feature engineering.
- No hyperparameter tuning.
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
    lolset = pd.read_csv("Datasets/ranked_10min_ds.csv")

    # Basic preprocessing
    # Drop ID column
    lolset = lolset.drop(columns=["gameId"])
    # Drop directly redundant features
    lolset = lolset.drop(columns=[
        "blueEliteMonsters", "blueAvgLevel", "blueCSPerMin", "blueGoldPerMin",
        "redFirstBlood", "redKills", "redDeaths", "redEliteMonsters",
        "redAvgLevel", "redCSPerMin", "redGoldPerMin", "redGoldDiff",
        "redExperienceDiff"
        ])
    # Initialize and train model with default hyperparameters
    lolmodel = LogReg()
    lolmodel.dataLoader(lolset, "blueWins")
    lolmodel.normalize()  # Z-score normalization

    # Train and evaluate model
    lolmodel.fit()
    lolmodel.evaluate()
    time.sleep(2)

    # Plot evaluation metrics
    lolmodel.convergencePlot("Outputs/redundant_Convergence.jpg")
    lolmodel.rocCurvePlot("Outputs/redundant_ROC.jpg")
    lolmodel.metricPlot(lolmodel.trainMetrics_,
                        "Outputs/redundant_train_Metrics.jpg")
    lolmodel.metricPlot(lolmodel.testMetrics_,
                        "Outputs/redundant_test_Metrics.jpg")
    lolmodel.confusionPlot(lolmodel.testConfusion_,
                           "Outputs/redundant_Confusion.jpg")
    time.sleep(3)

    # Print feature significance
    print("Model results:")
    cof = lolmodel.significantFeatures()
    print(cof[cof["PValue"] < 0.05])

    # Save model
    lolmodel.saveModel("Models/redundantModel.csv")
    lolset.corr().to_excel("Outputs/redundantCorrelations.xlsx")
    lolset.to_csv("Datasets/redundantDataset.csv", index=False)
    print("Model and data saved.")


if __name__ == "__main__":
    main()
