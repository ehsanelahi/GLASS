import json
import os
import platform
import sys
import time
import warnings
import pandas as pd
import numpy as np
import sklearn
import matplotlib
from sklearn.model_selection import train_test_split
from util import vecPredictProba
from CustomPlanner import CustomPlanner
from sklearn.neural_network import MLPClassifier
from analyze_results import summarize

# All paths are resolved relative to this file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUBJECT = "RR"
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "dataset5000.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results", SUBJECT)


# provided optimization score function (based on the ideal controllable feature assignment)
def optimizationScore(adaptation):
    return 400 - (100 - adaptation[0] + adaptation[1] + adaptation[2] + adaptation[3])



if __name__ == '__main__':

    programStartTime = time.time()

    # suppress all warnings
    warnings.filterwarnings("ignore")

    # Load dataset
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ds = pd.read_csv(DATASET_PATH)

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

    # One classifier per requirement, trained on the original (not oversampled) training split.
    models = []
    for i, req in enumerate(reqs):
        if i == 1:
            model = MLPClassifier(hidden_layer_sizes=(18,), random_state=42).fit(X_train, y_train[req])
        else:
            model = MLPClassifier(hidden_layer_sizes=(8,), random_state=42).fit(X_train, y_train[req])

        models.append(model)
        print("=" * 100)

    # Domains for each controllable feature (example: 0–100)
    controllableFeatureDomains = np.repeat([[0, 100]], n_controllableFeatures, axis=0)

    # Indices of controllable features
    controllableFeatureIndices = [0, 1, 2, 3]


    customPlanner = CustomPlanner(X=X_train, n_neighbors=n_neighbors, n_startingSolutions=n_startingSolutions,
                                    reqClassifiers=models, targetConfidence=targetConfidence,
                                    controllableFeaturesNames=controllableFeaturesNames,
                                    controllableFeatureIndices=controllableFeatureIndices,
                                    controllableFeatureDomains=controllableFeatureDomains,
                                    optimizationDirections=optimizationDirections,
                                    optimizationScoreFunction=optimizationScore,
                                    plotsPath=os.path.join(RESULTS_DIR, "explainability_plots"))


    # Define path for results CSV
    results_csv_path = os.path.join(RESULTS_DIR, "RR_custom.csv")

    # Initialize list to store results
    results = []


    testNum = 200
    initialConfidences = []
    for k in range(testNum):
        row = X_test.iloc[k, :].to_numpy()
        initialConfidences.append(vecPredictProba(models, [row])[0])

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

    # Machine-readable copy of the same results: one row per test case, one column
    # per requirement. 
    perInstance = pd.DataFrame({"test_instance": range(testNum),
                                "dataset_row": X_test.index[:testNum]})
    for j, req in enumerate(reqs):
        perInstance["initial_confidence_" + req] = [c[j] for c in initialConfidences]
    for j, name in enumerate(controllableFeaturesNames):
        perInstance["adaptation_" + name] = [r[0][controllableFeatureIndices[j]] for r in results]
    for j, req in enumerate(reqs):
        perInstance["confidence_" + req] = [r[1][j] for r in results]
    perInstance["score"] = [r[2] for r in results]
    perInstance["execution_time"] = [r[3] for r in results]
    perInstance.to_csv(os.path.join(RESULTS_DIR, "per_instance.csv"), index=False)

    runInfo = {
        "subject": SUBJECT,
        "dataset": os.path.relpath(DATASET_PATH, BASE_DIR).replace(os.sep, "/"),
        "n_train": int(X_train.shape[0]), "n_test": int(X_test.shape[0]), "n_evaluated": testNum,
        "confidence_threshold": float(targetConfidence[0]),
        "oversampling": "none",
        "selected_models": {req: "MLPClassifier(hidden_layer_sizes=%s, random_state=42)"
                                 % (m.hidden_layer_sizes,) for req, m in zip(reqs, models)},
        "planner_parameters": {name: getattr(customPlanner, name) for name in
                               ("numCounterfactuals", "maxIterations", "stepFactor",
                                "maxPossibilities", "topK", "refinementThreshold")},
        "versions": {"python": platform.python_version(), "platform": platform.platform(),
                     "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
        "total_runtime_seconds": time.time() - programStartTime,
    }
    with open(os.path.join(RESULTS_DIR, "run_info.json"), "w") as f:
        json.dump(runInfo, f, indent=2)

    summarize(os.path.join(RESULTS_DIR, "per_instance.csv"), float(targetConfidence[0]),
              os.path.join(RESULTS_DIR, "summary.csv"))
