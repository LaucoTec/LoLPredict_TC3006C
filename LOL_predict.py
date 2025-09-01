#Imports
import time
import pandas as pd
from LogisticRegression import LogReg



def main():
    """
    Main function for data analysis.
    """
    #Load data, clean columns
    lolset = pd.read_csv("ranked_10min_ds.csv")
    print(lolset.columns)
    print()
    lolset = lolset.drop(["gameId", "blueEliteMonsters", "blueTotalExperience", "blueTotalGold", "blueAvgLevel", "blueCSPerMin", "blueGoldPerMin", "redFirstBlood", "redKills", "redDeaths", "redEliteMonsters", "redTotalGold", "redAvgLevel", "redTotalExperience", "redGoldDiff", "redExperienceDiff", "redCSPerMin", "redGoldPerMin"], axis=1)
    print(lolset.columns)
    
    time.sleep(5)
    
    #Train or load a model
    lolmodel = LogReg(0.05, 1e-6, 10000, 0.5)
    lolmodel.dataLoader(lolset, "blueWins")
    
    opc = input("Would you like to load a model? [Y/N]")
    if opc  == "Y":
        lolmodel.loadModel(input("Write CSV filename"))
    else:
        lolmodel.fitModel()
        
    lolmodel.evaluate()
    
    #Menu for options
    while True:
        print("=====================")
        print("  Choose an option. ")
        print("=====================")
        print("1.- Metrics graph")
        print("2.- Convergence graph")
        print("3.- Confusion matrix")
        print("4.- ROC curve - AUC")
        print("5.- Save model")
        print("6.- Predict (test set)")
        print("7.- Exit")
        opc = int(input("Option: "))
        
        if opc == 1:
            lolmodel.metricPlot(lolmodel.trainMetrics_)
            lolmodel.metricPlot(lolmodel.testMetrics_)

        elif opc == 2:
            lolmodel.convergence()
            
        elif opc == 3:
            lolmodel.confusionPlot(lolmodel.trainConfussion_)
            lolmodel.confusionPlot(lolmodel.testConfussion_)
            
        elif opc == 4:
            lolmodel.rocCurve()
            
        elif opc == 5:
            lolmodel.saveModel("lolmodel")
            time.sleep(3)
            
        elif opc == 6:
            idx = int(input(f"Choose an index from 0 to {len(lolmodel.yTest)-1}. "))
            pred = lolmodel.predictClass(xTest[idx])
            
            if pred:
                print(f"Model predicted 1 (Win), real value is {lolmodel.yTest[idx]}")
            else:
                print(f"Model predicted 0 (Defeat), real value is {lolmodel.yTest[idx]}")
            time.sleep(3)
            
        elif opc == 7:
            break
        else:
            pass
    
if __name__ == "__main__":
    main()