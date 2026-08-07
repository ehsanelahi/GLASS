import os
import sys
import time
import warnings
import pandas as pd
import numpy as np
from colorama import Fore, Style
from sklearn.model_selection import train_test_split
from util import vecPredictProba
from model.ModelConstructor import constructModel

from CustomPlanner import CustomPlanner



def normalizeAdaptation(adaptation):
    new_adaptation = []
    for index in range(n_controllableFeatures):
        new_adaptation.append(((adaptation[index] - controllableFeatureDomains[index][0]) / (
                    controllableFeatureDomains[index][1] - controllableFeatureDomains[index][0])) * 100)

    return new_adaptation


# optimization score function (based on the ideal controllable feature assignment)
def optimizationScore(adaptation):
    adaptation = normalizeAdaptation(adaptation)
    score = 0
    tot = 100 * n_controllableFeatures
    for i in range(n_controllableFeatures):
        if optimizationDirections[i] == 1:
            score += 100 - adaptation[i]
        else:
            score += adaptation[i]
    score = score / tot
    return 1 - score


if __name__ == '__main__':

    programStartTime = time.time()


    # suppress all warnings
    warnings.filterwarnings("ignore")

    # Load dataset
    ds = pd.read_csv('dataset/uav.csv')

    # Define features and target columns
    featureNames = ['formation', 'flying_speed', 'countermeasure', 'weather', 'day_time', 'threat_range', '#threats'] #uav

    controllableFeaturesNames = featureNames[0:3]
    externalFeaturesNames = featureNames[3:7]
    
    # for simplicity, we consider all the ideal points to be 0 or 100
    # so that we just need to consider ideal directions instead
    # -1 => minimize, 1 => maximize

    # Optimization directions
    optimizationDirections = [1, 1, -1]

    reqs = ["req_0", "req_1", "req_2", "req_3", "req_4",
            "req_5", "req_6", "req_7", "req_8", "req_9", "req_10", "req_11"] #uav

    n_reqs = len(reqs)
    n_neighbors = 8
    n_startingSolutions = 5

    n_controllableFeatures = len(controllableFeaturesNames)
    targetConfidence = np.full((1, n_reqs), 0.8)[0]

    # split the dataset
    X = ds.loc[:, featureNames]
    y = ds.loc[:, reqs]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    models = []
    for req in reqs:
        print(Fore.RED + "Requirement: " + req + "\n" + Style.RESET_ALL)

        models.append(constructModel(X_train.values,
                                     X_test.values,
                                     np.ravel(y_train.loc[:, req]),
                                     np.ravel(y_test.loc[:, req])))
        print("=" * 100)

    # Domains for each controllable feature 
    controllableFeatureDomains = np.array([[0, 1], [5.0, 50.0], [0, 1]])

    # Indices of controllable features
    controllableFeatureIndices = [0, 1, 2]
    

    # Initialize planner
    
    customPlanner = CustomPlanner(X=X_train, n_neighbors=n_neighbors, n_startingSolutions=n_startingSolutions, 
                                    reqClassifiers=models, targetConfidence=targetConfidence, 
                                    controllableFeaturesNames=controllableFeaturesNames,
                                    controllableFeatureIndices=controllableFeatureIndices,
                                    controllableFeatureDomains=controllableFeatureDomains,
                                    optimizationDirections=optimizationDirections,
                                    optimizationScoreFunction= optimizationScore,
                                    plotsPath="explainability_plots/" )


    results_csv_path = "custom.csv"
    results = []

    testNum = 200
    for k in range(testNum):
        row = X_test.iloc[k, :].to_numpy()

        startTime = time.time()
        adaptation, confidence, score = customPlanner.findAdaptation(row)
        endTime = time.time()
        saceTime = endTime - startTime

        print("\nCustom algorithm execution time: " + str(saceTime) + " s")
        print("-" * 100)

        results.append([adaptation, confidence, score, saceTime])

    results_df = pd.DataFrame(results, columns=["adaptation", "confidence", "score", "execution_time"])
    results_df.to_csv(results_csv_path, index=False)

    print(f"\nResults saved to {results_csv_path}")


    sys.exit()

    
    