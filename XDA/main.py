import os
import sys
import time
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from util import vecPredictProba
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from CustomPlanner import CustomPlanner


# success score function (based on the signed distance with respect to the target success probabilities)
def successScore(adaptation, reqClassifiers, targetSuccessProba):
    return np.sum(vecPredictProba(reqClassifiers, [adaptation])[0] - targetSuccessProba)

def normalizeAdaptation(adaptation, domains, nFeatures):
    new_adaptation = []
    for index in range(nFeatures):
        new_adaptation.append(((adaptation[index] - domains[index][0]) / (domains[index][1] - domains[index][0])) * 100)

    return new_adaptation
    
# provided optimization score function (based on the ideal controllable feature assignment)
def optimizationScore(adaptation, domains, n_controllableFeatures):
    adaptation = normalizeAdaptation(adaptation, domains, n_controllableFeatures)
    return 200 - (100 - adaptation[0] + adaptation[1]) #driveDouble


if __name__ == '__main__':

    programStartTime = time.time()

    os.chdir(r"C:\Users\ehsan\Downloads\PDP\XDA")

    # suppress all warnings
    warnings.filterwarnings("ignore")

    # Load dataset
    ds = pd.read_csv('drivev3.csv')

    # Define features and target columns
    featureNames = ['car_speed','p_x','p_y','orientation','weather','road_shape'] #drive
    controllableFeaturesNames = featureNames[0:2]
    externalFeaturesNames = featureNames[2:6]
    controllableFeatureIndices = [0, 1]
    
    # for simplicity, we consider all the ideal points to be 0 or 100
    # so that we just need to consider ideal directions instead
    # -1 => minimize, 1 => maximize

    # Optimization directions
    optimizationDirections = [1, -1]  #drive

    reqs = ["req_0", "req_1", "req_2"] #drive

    n_reqs = len(reqs)
    n_neighbors = 8
    n_startingSolutions = 5

    n_controllableFeatures = len(controllableFeaturesNames)
    targetConfidence = np.full((1, n_reqs), 0.8)[0]

    # Map "TRUE" and "FALSE" to 1 and 0 in target columns
    X = ds.loc[:, featureNames]
    y = ds.loc[:, reqs].replace({True: 1, False: 0})

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

    models = []
    for req in reqs:
        #model = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=10).fit(X_train, y_train[req])
        model = MLPClassifier(hidden_layer_sizes=(28,), random_state=42).fit(X_train, y_train[req])
        models.append(model)

    # Domains for each controllable feature
    controllableFeatureDomains = np.array([[5.0, 50.0], [0.0, 10.0]])   #drive
    

    # Initialize planner
    

    customPlanner = CustomPlanner(X_train, n_neighbors, n_startingSolutions, models, targetConfidence,
                                  controllableFeaturesNames, controllableFeatureIndices, controllableFeatureDomains,
                                  optimizationDirections, lambda a: optimizationScore(a, controllableFeatureDomains, n_controllableFeatures), 1, "../explainability_plots")


    # Define path for results CSV
    results_csv_path = "drive_custom.csv"


    # Initialize list to store results
    results = []

    testNum = 100
    for k in range(testNum):
        row = X_test.iloc[k, :].to_numpy()
        print(f"\nTest {k + 1}: Row {k}")
        print(f"Original Row: {row}")

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

    
    