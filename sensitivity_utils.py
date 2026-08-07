import os
import time

import numpy as np
import pandas as pd


# Paper-reported default values for the six CustomPlanner1 hyperparameters that are
# "exposed for the sensitivity analysis" (see CustomPlanner1.__init__ docstring).
DEFAULT_PARAMS = {
    "numCounterfactuals": 3,
    "maxIterations": 10,
    "stepFactor": 0.25,
    "maxPossibilities": 10000,
    "topK": 20,
    "refinementThreshold": 0.8,
}

# One-at-a-time sweep grids. Each grid includes the default value so every
# per-parameter CSV contains a built-in reference row at the paper-default setting.
PARAM_GRIDS = {
    "numCounterfactuals": [1, 3, 5, 8, 10],
    "maxIterations": [2, 5, 10, 15, 20, 30],
    "stepFactor": [0.05, 0.1, 0.25, 0.5, 0.75, 1.0],
    "maxPossibilities": [100, 1000, 5000, 10000],
    "topK": [5, 10, 20, 50, 100, 200],
    "refinementThreshold": [0.5, 0.6, 0.7, 0.8, 0.9],
}

SUBJECTS = ["ADAS", "RR", "UAV"]

SUBJECT_REQS = {
    "ADAS": [f"req_{i}" for i in range(3)],
    "RR": [f"req_{i}" for i in range(4)],
    "UAV": [f"req_{i}" for i in range(12)],
}

# Derived from each subject's own optimizationScore function
# (main_drive.py / main_RR.py / main_UAV.py):
#   ADAS, UAV: 1 - normalized weighted distance -> bounded [0, 1]
#   RR:        400 - (100 - a0 + a1 + a2 + a3), a_i in [0, 100] -> [0, 400]
QUALITY_SCORE_RANGE = {
    "ADAS": (0.0, 1.0),
    "RR": (0.0, 400.0),
    "UAV": (0.0, 1.0),
}

CSV_COLUMNS = [
    "subject", "parameter_name", "parameter_value", "test_instance",
    "requirement", "satisfaction_likelihood", "success_rate",
    "quality_score", "execution_time",
]


N_TEST_INSTANCES = 200


def _evaluate_all_instances(customPlanner, X_test, reqs, targetConfidence,
                             n_test_instances=N_TEST_INSTANCES):
    """Run findAdaptation once per test instance under the planner's current
    hyperparameter settings, expanding each result into one row per requirement.

    Capped at n_test_instances (default 200) to match the existing evaluation
    pipeline's `testNum = 200` loop in main_drive.py/main_RR.py/main_UAV.py --
    X_test itself is a much larger held-out split (e.g. 2000 rows for ADAS);
    only its first 200 rows are ever evaluated there, so test_instance k here
    stays cross-referenceable with the existing *_custom.csv results."""
    rows = []
    n_test = min(n_test_instances, X_test.shape[0])

    for k in range(n_test):
        row = X_test.iloc[k, :].to_numpy()

        startTime = time.time()
        adaptation, confidence, score = customPlanner.findAdaptation(row)
        elapsed = time.time() - startTime

        for j, req in enumerate(reqs):
            if confidence is None:
                satisfaction_likelihood = np.nan
                success = 0
            else:
                satisfaction_likelihood = float(confidence[j])
                success = int(satisfaction_likelihood >= targetConfidence[j])

            rows.append({
                "test_instance": k,
                "requirement": req,
                "satisfaction_likelihood": satisfaction_likelihood,
                "success_rate": success,
                "quality_score": float(score) if score is not None else np.nan,
                "execution_time": elapsed,
            })

    return rows


def run_parameter_sweep(subject, customPlanner, X_test, reqs, targetConfidence,
                         param_name, values, output_dir="sensitivity_results"):
    """Sweep a single CustomPlanner1 hyperparameter over `values`, holding the other
    five at DEFAULT_PARAMS, reusing the already-built planner (models/KNN/PDPs are
    never recomputed). Writes sensitivity_results/{subject}/{param_name}.csv."""
    all_rows = []

    for value in values:
        # Reset every sensitivity parameter to its default, then override only the
        # one under test -- this is what keeps the sweep one-at-a-time.
        for name, default_value in DEFAULT_PARAMS.items():
            setattr(customPlanner, name, default_value)
        setattr(customPlanner, param_name, value)

        instance_rows = _evaluate_all_instances(customPlanner, X_test, reqs, targetConfidence)
        for r in instance_rows:
            r["subject"] = subject
            r["parameter_name"] = param_name
            r["parameter_value"] = value
        all_rows.extend(instance_rows)

    # Leave the planner at default settings for whatever runs next.
    for name, default_value in DEFAULT_PARAMS.items():
        setattr(customPlanner, name, default_value)

    df = pd.DataFrame(all_rows, columns=CSV_COLUMNS)

    subject_dir = os.path.join(output_dir, subject)
    os.makedirs(subject_dir, exist_ok=True)
    csv_path = os.path.join(subject_dir, f"{param_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[{subject}] {param_name}: wrote {len(df)} rows to {csv_path}")

    return df


def run_sensitivity_analysis(subject, customPlanner, X_test, reqs, targetConfidence,
                              param_grids=None, output_dir="sensitivity_results"):
    """Sweep all six CustomPlanner1 sensitivity parameters, one at a time, for a
    single case-study subject. Reuses the caller's already-trained models and
    already-built customPlanner (offline PDP/KNN preprocessing is never redone)."""
    if param_grids is None:
        param_grids = PARAM_GRIDS

    print(f"\n{'=' * 100}\nRunning parameter sensitivity analysis for subject: {subject}\n{'=' * 100}")

    results = {}
    for param_name, values in param_grids.items():
        startTime = time.time()
        results[param_name] = run_parameter_sweep(
            subject, customPlanner, X_test, reqs, targetConfidence,
            param_name, values, output_dir=output_dir,
        )
        print(f"[{subject}] {param_name}: completed in {time.time() - startTime:.2f} s")

    return results
