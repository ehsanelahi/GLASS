

**GLASS** is a Python-based planning and adaptation framework that uses
machine learning, counterfactual reasoning, and partial dependence plots to find
optimal adaptations that satisfy multiple system requirements.

The project combines:
- Supervised learning
- Counterfactual generation
- Partial Dependence Plots (PDPs)
- Multi-objective optimization

---

## 🚀 Key Features

- 🤖 Requirement satisfaction using trained ML classifiers
- 🔁 Counterfactual-based adaptation generation
- 📊 Refinement via Partial Dependence Plots (PDPs)
- 🎯 Optimization-driven adaptation selection
- 🧠 Supports multiple requirements simultaneously

---

## 🛠️ Technologies Used

- **Python**
- **NumPy & Pandas** – data processing
- **Scikit-learn** – ML models & KNN
- **Matplotlib / Seaborn** – explainability plots

---

## 📂 Project Structure
```text

GLASS/
├── main.py                     # Entry point
├── CustomPlanner.py             # Core planning logic
├── model/
│   └── ModelConstructor.py     # ML model creation
├── util.py                     # Utility functions
├── explainability_techniques/
│   └── PDP.py                  # Partial Dependence Plots
├── uav.csv              # Dataset
├── requirements.txt
└── README.md

```
---

##  Install dependencies
pip install -r requirements.txt

---

## ▶️ Usage
python main.py
```text
This will:
  Train ML models for each requirement
  Generate explainability plots
  Search for optimal adaptations
  Save results to a CSV file
```
---

## 📊 Output
```text
Explainability plots saved under:
explainability_plots/

Sensitivity plots saved under:
sensitivity_plots/

Adaptation results saved as:
custom.csv

```
