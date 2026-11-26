"""Aggregate the per-instance results into the paper's metrics.

Metric 1 satisfaction likelihood
after adaptation -- reported here as median and quartiles over all test cases.
Metric 2 success rate

Usage:  python analyze_results.py [results/UAV/per_instance.csv] [tau]
"""
import os
import sys

import pandas as pd


def summarize(per_instance_csv, tau=0.8, out_csv=None):
    df = pd.read_csv(per_instance_csv)
    reqs = [c[len("confidence_"):] for c in df.columns if c.startswith("confidence_")]
    rows = []
    for i, req in enumerate(reqs):
        conf = df["confidence_" + req]
        init = df["initial_confidence_" + req]
        rows.append({
            "requirement": "R%d" % (i + 1),
            "column": req,
            "n_instances": len(df),
            "success_rate": (conf > tau).mean(),
            "success_rate_before_adaptation": (init > tau).mean(),
            "satisfaction_likelihood_q1": conf.quantile(0.25),
            "satisfaction_likelihood_median": conf.median(),
            "satisfaction_likelihood_q3": conf.quantile(0.75),
        })
    summary = pd.DataFrame(rows)
    print("\nPer-requirement results over %d test cases (tau = %s)" % (len(df), tau))
    print(summary.drop(columns=["column"]).round(4).to_string(index=False))
    print("\nAll requirements satisfied simultaneously: %.4f"
          % (df[["confidence_" + r for r in reqs]] > tau).all(axis=1).mean())
    #print("Median score: %.4f   Median execution time: %.4f s"
          #% (df["score"].median(), df["execution_time"].median()))
    if out_csv is not None:
        summary.to_csv(out_csv, index=False)
        print("Summary saved to " + out_csv)
    return summary


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "results", "UAV", "per_instance.csv")
    tau = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    summarize(path, tau, os.path.join(os.path.dirname(path), "summary.csv"))
