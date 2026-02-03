

**GLASS** is a Python-based intelligent planning and adaptation framework that uses
machine learning, counterfactual reasoning, and explainability techniques to find
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
- 📊 Explainability via Partial Dependence Plots (PDP)
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
├── dataset5000.csv              # Dataset
├── requirements.txt
└── README.md

---

##  Install dependencies
pip install -r requirements.txt

---

## Usage
python main.py

This will:
  Train ML models for each requirement
  Generate explainability plots
  Search for optimal adaptations
  Save results to a CSV file

---

## Output
Explainability plots saved under:
explainability_plots/

Adaptation results saved as:
custom.csv
