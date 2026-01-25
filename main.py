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
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import BorderlineSMOTE, SVMSMOTE



# provided optimization score function (based on the ideal controllable feature assignment)
def optimizationScore(adaptation):
    return 400 - (100 - adaptation[0] + adaptation[1] + adaptation[2] + adaptation[3])



if __name__ == '__main__':

    programStartTime = time.time()

    #os.chdir(r"C:\Users\ehsan\Downloads\PDP")

    # suppress all warnings
    warnings.filterwarnings("ignore")

    # Load dataset
    ds = pd.read_csv('dataset5000.csv')

    # Define features and target columns
    featureNames = ["cruise speed", "image resolution", "illuminance", "controls responsiveness",
                    "power", "smoke intensity", "obstacle size", "obstacle distance", "firm obstacle"]

    controllableFeaturesNames = featureNames[0:4]
    externalFeaturesNames = featureNames[4:9]
    
    # for simplicity, we consider all the ideal points to be 0 or 100
    # so that we just need to consider ideal directions instead
    # -1 => minimize, 1 => maximize

    # Optimization directions
    optimizationDirections = [1, -1, -1, -1]

    reqs = ["req_0", "req_1", "req_2", "req_3"]

    n_reqs = len(reqs)
    n_neighbors = 5
    n_startingSolutions = 5

    n_controllableFeatures = len(controllableFeaturesNames)
    targetConfidence = np.full((1, n_reqs), 0.8)[0]

    # split the dataset
    X = ds.loc[:, featureNames]
    y = ds.loc[:, reqs]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    

    models = []

    for req in reqs:
        print(Fore.RED + "Requirement: " + req + "\n" + Style.RESET_ALL)

        models.append(constructModel(X_train.values,
                                     X_test.values,
                                     np.ravel(y_train.loc[:, req]),
                                     np.ravel(y_test.loc[:, req])))
        print("=" * 100)

    # Domains for each controllable feature (example: 0–100)
    controllableFeatureDomains = np.repeat([[0, 100]], n_controllableFeatures, axis=0)

    # Indices of controllable features
    controllableFeatureIndices = [0, 1, 2, 3]

    # Initialize planner
    
    
    customPlanner = CustomPlanner(X=X_train, n_neighbors=n_neighbors, n_startingSolutions=n_startingSolutions, 
                                    reqClassifiers=models, targetConfidence=targetConfidence, 
                                    controllableFeaturesNames=controllableFeaturesNames,
                                    controllableFeatureIndices=controllableFeatureIndices,
                                    controllableFeatureDomains=controllableFeatureDomains,
                                    optimizationDirections=optimizationDirections,
                                    optimizationScoreFunction=optimizationScore,
                                    plotsPath="explainability_plots/RR/" )


    # Define path for results CSV
    results_csv_path = "RR_custom.csv"


    # Initialize list to store results
    results = []
    

    testNum = 200
    for k in range(testNum):
        row = X_test.iloc[k, :].to_numpy()
        #print(f"\nTest {k + 1}: Row {k}")
        #print(f"Original Row: {row}")

        startTime = time.time()
        adaptation, confidence, score = customPlanner.findAdaptation(row)
        #adaptation, confidence, score = sacePlanner.findAdaptation(row)
        endTime = time.time()
        saceTime = endTime - startTime

        print("\nCustom algorithm execution time: " + str(saceTime) + " s")
        print("-" * 100)

        results.append([adaptation, confidence, score, saceTime])

    results_df = pd.DataFrame(results, columns=["adaptation", "confidence", "score", "execution_time"])
    results_df.to_csv(results_csv_path, index=False)

    print(f"\nResults saved to {results_csv_path}")

    sys.exit()

    
    